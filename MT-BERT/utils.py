import contextlib
import sys
from tqdm.contrib import DummyTqdmFile
import torch
import os

@contextlib.contextmanager
def stream_redirect_tqdm():
    orig_out_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = map(DummyTqdmFile, orig_out_err)
        yield orig_out_err[0]
    # Relay exceptions
    except Exception as exc:
        raise exc
    # Always restore sys.stdout/err if necessary
    finally:
        sys.stdout, sys.stderr = orig_out_err




def save_prompts(model, output_dir, attn_prefix_tuning, shared_attn, num_target, task_name):
    for name, param in model.named_parameters():
        # Save prompt weights.
        if attn_prefix_tuning is False and ("prefix_shared" in name or "prefix" in name):
            shared_params = param
            torch.save(shared_params, os.path.join(
                output_dir, "prefix_embeddings.pt"))
        elif attn_prefix_tuning is True and name == "prefix_shared":
            shared_params = param
            if shared_attn is True:
                for i in range(num_target):
                    torch.save(shared_params[i], os.path.join(
                        output_dir, "prefix_embeddings_{}.pt".format(task_name[i])))
            else:
                torch.save(shared_params, os.path.join(
                    output_dir, "prefix_embeddings.pt"))

        # Save attention and layer norm weights.
        if attn_prefix_tuning is True and "encoder.attn_Wa.weight" == name:
            attn_weights_params = param
            torch.save(attn_weights_params, os.path.join(
                output_dir, "attn_Wa_weights.pt"))
        if attn_prefix_tuning is True and "encoder.attn_W_down.weight" == name:
            attn_weights_params = param
            torch.save(attn_weights_params, os.path.join(
                output_dir, "attn_W_down.pt"))
        if attn_prefix_tuning is True and "encoder.attn_W_up.weight" == name:
            attn_weights_params = param
            torch.save(attn_weights_params, os.path.join(
                output_dir, "attn_W_up.pt"))
        if attn_prefix_tuning is True and "encoder.layer_norm.weight" == name:
            attn_weights_params = param
            torch.save(attn_weights_params, os.path.join(
                output_dir, "layer_norm_weight.pt"))
        if attn_prefix_tuning is True and "encoder.layer_norm.bias" == name:
            attn_weights_params = param
            torch.save(attn_weights_params, os.path.join(
                output_dir, "layer_norm_bias.pt"))