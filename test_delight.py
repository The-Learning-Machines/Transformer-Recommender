import torch
from delight import DeLighTTransformer, default_arg


default_arg.delight_enc_max_depth = 4
default_arg.delight_enc_min_depth = 2
default_arg.delight_enc_layers = 4
default_arg.delight_enc_width_mult = 2.0

model = DeLighTTransformer(128, 512, 120, args=default_arg)
inp = torch.randint(0, 512, (32, 100))

out = model(inp)
print(out, out.shape)


# delight_dropout
# delight_enc_max_groups
# act_type
# norm_type
# glt_shuffle
# define_iclr
# attention_dropout
# dropout
# encoder_normalize_before
# ffn_dropout
# delight_enc_ffn_red
