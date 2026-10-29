# Modified from S-FLM (https://github.com/jdeschena/s-flm) for BCT: temperature conditioning in `DDiTBlock` / `DDiTFinalLayer`.
import math
import typing

import einops
import flash_attn
import flash_attn.layers.rotary
import huggingface_hub
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F

# Flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)


def bias_dropout_add_scale(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float,
    training: bool) -> torch.Tensor:
  if bias is not None:
    out = scale * F.dropout(x + bias, p=prob, training=training)
  else:
    out = scale * F.dropout(x, p=prob, training=training)

  if residual is not None:
    out = residual + out
  return out


def get_bias_dropout_add_scale(training):
  def _bias_dropout_add(x, bias, scale, residual, prob):
    return bias_dropout_add_scale(
      x, bias, scale, residual, prob, training)

  return _bias_dropout_add


# function overload
def modulate(x: torch.Tensor,
             shift: torch.Tensor,
             scale: torch.Tensor) -> torch.Tensor:
  return x * (1 + scale) + shift


@torch.jit.script
def bias_dropout_add_scale_fused_train(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float) -> torch.Tensor:
  return bias_dropout_add_scale(
    x, bias, scale, residual, prob, True)


@torch.jit.script
def bias_dropout_add_scale_fused_inference(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float) -> torch.Tensor:
  return bias_dropout_add_scale(
    x, bias, scale, residual, prob, False)


@torch.jit.script
def modulate_fused(x: torch.Tensor,
                   shift: torch.Tensor,
                   scale: torch.Tensor) -> torch.Tensor:
  return modulate(x, shift, scale)


class Rotary(torch.nn.Module):
  def __init__(self, dim, base=10_000):
    super().__init__()
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    self.register_buffer('inv_freq', inv_freq)
    self.seq_len_cached = None
    self.cos_cached = None
    self.sin_cached = None

  def _compute_cos_sin(self, t: torch.Tensor):
    freqs = torch.einsum("i,j->ij", t, self.inv_freq.clone())
    emb = torch.cat((freqs, freqs), dim=-1).to(t.device)
    cos = emb.cos()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
    sin = emb.sin()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
    cos[:, :, 2, :, :].fill_(1.)
    sin[:, :, 2, :, :].fill_(0.)
    return cos, sin

  def forward(self, x, seq_dim=1):
    seq_len = x.shape[seq_dim]
    if seq_len != self.seq_len_cached:
      self.seq_len_cached = seq_len
      t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
      self.cos_cached, self.sin_cached = self._compute_cos_sin(t)
    return self.cos_cached, self.sin_cached

  def forward_range(self, start_idx: int, end_idx: int, device=None):
    if device is None:
      device = self.inv_freq.device
    t = torch.arange(start_idx, end_idx, device=device).type_as(self.inv_freq)
    return self._compute_cos_sin(t)


def rotate_half(x):
  x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
  return torch.cat((-x2, x1), dim=-1)


def split_and_apply_rotary_pos_emb(qkv, rotary_cos_sin):
  with torch.amp.autocast('cuda', enabled=False):
    cos, sin = rotary_cos_sin
    cos = cos.to(qkv.dtype)
    sin = sin.to(qkv.dtype)
    cos = cos[0,:,0,0,:cos.shape[-1]//2]
    sin = sin[0,:,0,0,:sin.shape[-1]//2]
    q, k, v = qkv.chunk(3, dim=2)
    q = flash_attn.layers.rotary.apply_rotary_emb_torch(
      q.squeeze(dim=2), cos, sin)
    k = flash_attn.layers.rotary.apply_rotary_emb_torch(
      k.squeeze(dim=2), cos, sin)
    v = v.squeeze(dim=2)
  return q, k, v


def apply_rotary_pos_emb(qkv, cos, sin):
  cos = cos[0,:,0,0,:cos.shape[-1]//2]
  sin = sin[0,:,0,0,:sin.shape[-1]//2]
  return flash_attn.layers.rotary.apply_rotary_emb_qkv_(qkv, cos, sin)


def regular_attention_multi_headed(q, k, v):
  # Assuming qkv is a tensor with shape [batch, seq_len, 3, num_heads, head_dim]
  # where the 3 represents Q, K, V packed in that order
  attention_output = F.scaled_dot_product_attention(
    query=q.transpose(1, 2),
    key=k.transpose(1, 2),
    value=v.transpose(1, 2),
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False)
  # [batch_size, seq_len, num_heads, head_dim]
  attention_output = attention_output.transpose(1, 2)
  return einops.rearrange(attention_output, 'b s h d -> b s (h d)')


def _sdpa_full(q, k, v):
  """Non-causal full-sequence attention. q/k/v: [B, H, S, D]."""
  return F.scaled_dot_product_attention(q, k, v)


#################################################################################
#                                  Layers                                       #
#################################################################################
class LayerNorm(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.weight = nn.Parameter(torch.ones([dim]))
    self.dim = dim
  def forward(self, x):
    with torch.amp.autocast('cuda', enabled=False):
      x = F.layer_norm(x.float(), [self.dim])
    return x * self.weight[None, None, :]


def residual_linear(x, W, x_skip, residual_scale):
  """x_skip + residual_scale * W @ x"""
  dim_out, dim_in = W.shape[0], W.shape[1]
  return torch.addmm(
    x_skip.view(-1, dim_out),
    x.view(-1, dim_in),
    W.T,
    alpha=residual_scale).view(*x.shape[:-1], dim_out)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################
class TimestepEmbedder(nn.Module):
  """
  Embeds scalar timesteps into vector representations.
  """
  def __init__(self, hidden_size, frequency_embedding_size=256):
    super().__init__()
    self.mlp = nn.Sequential(
      nn.Linear(frequency_embedding_size, hidden_size, bias=True),
      nn.SiLU(),
      nn.Linear(hidden_size, hidden_size, bias=True))
    self.frequency_embedding_size = frequency_embedding_size

  @staticmethod
  def timestep_embedding(t, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
    half = dim // 2
    freqs = torch.exp(
      - math.log(max_period)
      * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
      / half)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
      embedding = torch.cat(
        [embedding,
         torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

  def forward(self, t):
    t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
    t_emb = self.mlp(t_freq)
    return t_emb


class LabelEmbedder(nn.Module):
  """Embeds class labels into vector representations.
  
  Also handles label dropout for classifier-free guidance.
  """
  def __init__(self, num_classes, cond_size):
    super().__init__()
    self.embedding_table = nn.Embedding(num_classes + 1, cond_size)
    self.num_classes = num_classes

    # TODO think of initializing with 0.02 std deviation like in original DiT paper

  def forward(self, labels):
    embeddings = self.embedding_table(labels)
    return embeddings
    

#################################################################################
#                                 Core Model                                    #
#################################################################################

class DDiTBlockCausal(nn.Module):
  def __init__(self, dim, n_heads, adaLN: bool,
               cond_dim=None, mlp_ratio=4, dropout=0.1):
    super().__init__()
    self.n_heads = n_heads
    self.adaLN = adaLN

    self.norm1 = LayerNorm(dim)
    self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
    self.attn_out = nn.Linear(dim, dim, bias=False)
    self.dropout1 = nn.Dropout(dropout)

    self.norm2 = LayerNorm(dim)
    self.mlp = nn.Sequential(
      nn.Linear(dim, mlp_ratio * dim, bias=True),
      nn.GELU(approximate='tanh'),
      nn.Linear(mlp_ratio * dim, dim, bias=True))
    self.dropout2 = nn.Dropout(dropout)
    self.dropout = dropout

    if self.adaLN:
      self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim)
      self.adaLN_modulation.weight.data.zero_()
      self.adaLN_modulation.bias.data.zero_()
    self.k_cache = None
    self.v_cache = None

  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return bias_dropout_add_scale_fused_inference

  def reset_kv_cache(self):
    self.k_cache = None
    self.v_cache = None

  def forward(self, x, rotary_cos_sin, c=None, *, kv_cache=False):
    batch_size, seq_len = x.shape[0], x.shape[1]

    bias_dropout_scale_fn = self._get_bias_dropout_scale()

    # attention operation
    x_skip = x
    x = self.norm1(x)

    if self.adaLN:
      (shift_msa, scale_msa, gate_msa, shift_mlp,
       scale_mlp, gate_mlp) = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
      x = modulate_fused(x, shift_msa, scale_msa)

    qkv = self.attn_qkv(x)
    qkv = einops.rearrange(
      qkv, 'b s (three h d) -> b s three h d', three=3, h=self.n_heads)
    with torch.amp.autocast('cuda', enabled=False):
      cos, sin = rotary_cos_sin
      qkv = apply_rotary_pos_emb(qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
    if kv_cache:
      k_full = k if self.k_cache is None else torch.cat([self.k_cache, k], dim=1)
      v_full = v if self.v_cache is None else torch.cat([self.v_cache, v], dim=1)
      x = flash_attn.flash_attn_func(q, k_full, v_full, causal=True)
      self.k_cache = k.detach() if self.k_cache is None else torch.cat([self.k_cache, k.detach()], dim=1)
      self.v_cache = v.detach() if self.v_cache is None else torch.cat([self.v_cache, v.detach()], dim=1)
    else:
      x = flash_attn.flash_attn_func(q, k, v, causal=True)
    x = einops.rearrange(x, 'b s h d -> b s (h d)')

    if self.adaLN:
      x = bias_dropout_scale_fn(
        self.attn_out(x), None, gate_msa, x_skip, self.dropout)
      x = bias_dropout_scale_fn(
        self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp)),
        None, gate_mlp, x, self.dropout)
      return x
    else:
      scale = torch.ones(1, device=x.device, dtype=x.dtype)
      x = bias_dropout_scale_fn(
        self.attn_out(x), None, scale, x_skip, self.dropout)
      x = bias_dropout_scale_fn(
        self.mlp(self.norm2(x)), None, scale, x, self.dropout)
      return x



class DDiTBlock(nn.Module):
  def __init__(self, dim, n_heads, adaLN,
               cond_dim=None, mlp_ratio=4,
               dropout=0.1, softcap=0.0, temp_conditioning=False):
    super().__init__()
    self.n_heads = n_heads
    self.adaLN = adaLN
    self.softcap = softcap

    self.norm1 = LayerNorm(dim)
    self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
    self.attn_out = nn.Linear(dim, dim, bias=False)
    self.dropout1 = nn.Dropout(dropout)

    self.norm2 = LayerNorm(dim)
    self.mlp = nn.Sequential(
      nn.Linear(dim, mlp_ratio * dim, bias=True),
      nn.GELU(approximate='tanh'),
      nn.Linear(mlp_ratio * dim, dim, bias=True))
    self.dropout2 = nn.Dropout(dropout)
    self.dropout = dropout

    if self.adaLN:
      self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim)
      self.adaLN_modulation.weight.data.zero_()
      self.adaLN_modulation.bias.data.zero_()

    # TFT temperature conditioning: a zero-initialised modulation of the adaLN
    # parameters, driven by an embedding of tau = -log T and multiplied by tau.
    self.temp_conditioning = temp_conditioning
    if temp_conditioning:
      if not self.adaLN:
        raise ValueError(
          'temp_conditioning rides on the adaLN modulation, which is off.')
      self.temp_modulation = nn.Linear(cond_dim, 6 * dim)
      self.temp_modulation.weight.data.zero_()
      self.temp_modulation.bias.data.zero_()

    self.k_cache = None
    self.v_cache = None

  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return bias_dropout_add_scale_fused_inference

  def reset_kv_cache(self):
    self.k_cache = None
    self.v_cache = None

  def _attn(self, qkv: torch.Tensor, attn_kernel) -> torch.Tensor:
    """Standard full-sequence attention. qkv: [B, S, 3, H, D]"""
    if self.softcap > 0 and attn_kernel is _sdpa_full:
      x = flash_attn.flash_attn_qkvpacked_func(
        qkv, 0.0, causal=False, softcap=self.softcap)
      return einops.rearrange(x, 'b s h d -> b s (h d)')
    qkv = einops.rearrange(qkv, 'b s three h d -> b h three s d')
    x = attn_kernel(qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2])
    return einops.rearrange(x, 'b h s d -> b s (h d)')

  def _attn_kv_cache(self, qkv: torch.Tensor, attn_kernel,
                     cache_commit_len: int) -> torch.Tensor:
    """Attention with KV cache. qkv: [B, S_new, 3, H, D].

    Prepends cached K/V before attending, then commits the first
    cache_commit_len new tokens to the cache.
    """
    qkv = einops.rearrange(qkv, 'b s three h d -> b h three s d')
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
    k_full = k if self.k_cache is None else torch.cat([self.k_cache, k], dim=2)
    v_full = v if self.v_cache is None else torch.cat([self.v_cache, v], dim=2)
    x = attn_kernel(q, k_full, v_full)
    if cache_commit_len > 0:
      k_c = k[:, :, :cache_commit_len].detach()
      v_c = v[:, :, :cache_commit_len].detach()
      self.k_cache = k_c if self.k_cache is None else torch.cat([self.k_cache, k_c], dim=2)
      self.v_cache = v_c if self.v_cache is None else torch.cat([self.v_cache, v_c], dim=2)
    return einops.rearrange(x, 'b h s d -> b s (h d)')

  def forward(self, x, rotary_cos_sin, c=None, *,
              attn_kernel=None, cache_commit_len=None, kv_cache=False,
              temp_mod=None):
    bias_dropout_scale_fn = self._get_bias_dropout_scale()

    x_skip = x
    x = self.norm1(x)

    if self.adaLN:
      mod = self.adaLN_modulation(c)
      if temp_mod is not None:
        # temp_mod = (tau [B, 1], T_emb [B, cond]).
        tau, T_emb = temp_mod
        mod = mod + tau * self.temp_modulation(T_emb)
      (shift_msa, scale_msa, gate_msa, shift_mlp,
       scale_mlp, gate_mlp) = mod[:, None].chunk(6, dim=2)
      x = modulate_fused(x, shift_msa, scale_msa)

    qkv = einops.rearrange(
      self.attn_qkv(x), 'b s (three h d) -> b s three h d', three=3, h=self.n_heads)
    with torch.amp.autocast('cuda', enabled=False):
      cos, sin = rotary_cos_sin
      qkv = apply_rotary_pos_emb(qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))

    _kernel = attn_kernel if attn_kernel is not None else _sdpa_full
    if cache_commit_len is None:
      x = self._attn(qkv, _kernel)
    else:
      x = self._attn_kv_cache(qkv, _kernel, cache_commit_len)

    if self.adaLN:
      x = bias_dropout_scale_fn(self.attn_out(x), None, gate_msa, x_skip, self.dropout)
      x = bias_dropout_scale_fn(
        self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp)),
        None, gate_mlp, x, self.dropout)
    else:
      scale = torch.ones(1, device=x.device, dtype=x.dtype)
      x = bias_dropout_scale_fn(self.attn_out(x), None, scale, x_skip, self.dropout)
      x = bias_dropout_scale_fn(self.mlp(self.norm2(x)), None, scale, x, self.dropout)
    return x


class EmbeddingLayer(nn.Module):
  def __init__(self, dim, vocab_dim):
    super().__init__()
    self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
    torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

  def forward(self, x):
    if x.ndim == 2:
      return self.embedding[x]
    assert x.ndim == 3
    return torch.einsum(
      "blv,ve->ble",
      torch.nn.functional.softmax(x, dim=-1).float(),
      self.embedding.float()).to(x.dtype)


class DDiTFinalLayer(nn.Module):
  def __init__(self, hidden_size, out_channels, cond_dim,
               adaLN, temp_conditioning=False):
    super().__init__()
    self.norm_final = LayerNorm(hidden_size)
    self.linear = nn.Linear(hidden_size, out_channels)
    self.linear.weight.data.zero_()
    self.linear.bias.data.zero_()
    self.adaLN = adaLN
    if self.adaLN:
      self.adaLN_modulation = nn.Linear(cond_dim,
                                        2 * hidden_size,
                                        bias=True)
      self.adaLN_modulation.weight.data.zero_()
      self.adaLN_modulation.bias.data.zero_()
    # Same zero-initialised temperature modulation as in `DDiTBlock`.
    self.temp_conditioning = temp_conditioning
    if temp_conditioning:
      if not self.adaLN:
        raise ValueError(
          'temp_conditioning rides on the adaLN modulation, which is off.')
      self.temp_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
      self.temp_modulation.weight.data.zero_()
      self.temp_modulation.bias.data.zero_()


  def forward(self, x, c, temp_mod=None):
    x = self.norm_final(x)
    if self.adaLN:
      mod = self.adaLN_modulation(c)
      if temp_mod is not None:
        tau, T_emb = temp_mod
        mod = mod + tau * self.temp_modulation(T_emb)
      shift, scale = mod[:, None].chunk(2, dim=2)
      x = modulate_fused(x, shift, scale)
    x = self.linear(x)
    return x


class DIT(nn.Module, huggingface_hub.PyTorchModelHubMixin):
  def __init__(self, config, vocab_size: int):
    super().__init__()
    if type(config) == dict:
      config = omegaconf.OmegaConf.create(config)
    self.causal = config.algo.causal_attention
    self.adaLN = config.algo.adaLN
    self.config = config
    self.vocab_size = vocab_size
    dim = config.model.hidden_size
    cond_dim = config.model.cond_dim
    self.vocab_embed = EmbeddingLayer(dim, vocab_size)
    if self.adaLN:
      self.sigma_map = TimestepEmbedder(cond_dim)
    self.rotary_emb = Rotary(dim // config.model.n_heads)

    softcap = getattr(config.model, 'softcap', 0.0)
    blocks = []
    for _ in range(config.model.n_blocks):
      if self.causal:
        block = DDiTBlockCausal(
          dim=dim,
          n_heads=config.model.n_heads,
          cond_dim=cond_dim,
          adaLN=self.adaLN,
          dropout=config.model.dropout)
      else:
        block = DDiTBlock(
          dim=dim,
          n_heads=config.model.n_heads,
          cond_dim=cond_dim,
          adaLN=self.adaLN,
          dropout=config.model.dropout,
          softcap=softcap)
      blocks.append(block)
    self.blocks = nn.ModuleList(blocks)

    self.output_layer = DDiTFinalLayer(
      hidden_size=dim,
      out_channels=vocab_size,
      cond_dim=cond_dim,
      adaLN=self.adaLN)
    self.ctx_cached_len = 0

  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return  bias_dropout_add_scale_fused_inference

  def reset_kv_cache(self):
    self.ctx_cached_len = 0
    for block in self.blocks:
      block.reset_kv_cache()

  def forward(self, x0, xt, sigma, context=None):
    del x0
    kv_cache = False if context is None else context.kv_cache
    x = self.vocab_embed(xt)
    if self.adaLN:
      t_cond = F.silu(self.sigma_map(sigma))
    else:
      t_cond = None

    if kv_cache:
      rotary_cos_sin = self.rotary_emb.forward_range(
        self.ctx_cached_len, self.ctx_cached_len + x.shape[1],
        device=x.device)
    else:
      rotary_cos_sin = self.rotary_emb(x)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
      for i in range(len(self.blocks)):
        x = self.blocks[i](x, rotary_cos_sin, c=t_cond, kv_cache=kv_cache)
      x = self.output_layer(x, c=t_cond)

    if kv_cache:
      self.ctx_cached_len += xt.shape[1]
    return x
