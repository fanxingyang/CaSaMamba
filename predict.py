import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from dna_map import read_dna_sequences, map_to_indices
from net.model_args import ModelArgs
import numpy as np
import os
import torch

batch_size = 64
model_name = os.getenv("CASAMAMBA_MODEL", "b0")
dataset_name = os.getenv("CASAMAMBA_DATASET", "6mA_C.elegans")


def build_model(model_args):
    if model_name == "b0":
        from net.models.b0_mamba6ma import Mamba
    elif model_name == "b1":
        from net.models.b1_mamba2 import Mamba
    elif model_name == "b2":
        from net.models.b2_centeraware import Mamba
    else:
        raise ValueError("Unsupported model_name, expected one of: b0, b1, b2")
    return Mamba(model_args)


def infer_dataset_name(path):
    return os.path.basename(os.path.dirname(path))


def resolve_model_path(model_path, test_path):
    if model_path:
        return model_path
    dataset_name = infer_dataset_name(test_path)
    return os.path.join("checkpoints", dataset_name, f"best_{model_name}.pth")


def resolve_output_path(output_excel_path, test_path):
    if output_excel_path:
        return output_excel_path
    dataset_name = infer_dataset_name(test_path)
    os.makedirs("result", exist_ok=True)
    return os.path.join("result", f"output_logits_{dataset_name}_{model_name}.xlsx")


def test_main(test_path, model_path, output_excel_path):

    test_file_path = test_path
    model_path = resolve_model_path(model_path, test_file_path)
    output_excel_path = resolve_output_path(output_excel_path, test_file_path)
    test_sequences = read_dna_sequences(test_file_path)

    print("Encoding test sequences", flush=True)
    test_encoded = map_to_indices(test_sequences)

    X_test = np.array(test_encoded).astype(np.int64)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    test_dataset = TensorDataset(X_test_tensor)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model_args = ModelArgs()
    model_args.__post_init__()
    mamba_model = build_model(model_args)


    mamba_model.load_state_dict(torch.load(model_path))
    mamba_model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mamba_model.to(device)

    logits_list = []

    with torch.no_grad():
        for inputs in tqdm(test_dataloader, desc='Testing'):
            inputs = inputs[0].to(device)
            batch_logits = mamba_model(inputs)
            logits_list.extend(batch_logits.cpu().numpy())


    logits_df = pd.DataFrame(logits_list, columns=['Logits'])
    logits_df.to_excel(output_excel_path, index=False)
    print(f'Logits saved to {output_excel_path}')
    print(f'Model loaded from {model_path}')

if __name__ == "__main__":
    test_main(
        f'./data/{dataset_name}/test_pos.txt',
        '',
        ''
    )
