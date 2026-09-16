"""
BART autoencoder baseline for semantic compression of first-order logic premises.

Passes the FOL premises of the FOLIO dataset through a linear bottleneck placed
between a BART encoder and decoder, then reconstructs them. This is the learned
compression baseline the semantic framework is compared against: it compresses
effectively but gives no account of which logical content survives the bottleneck.
"""

import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BartForConditionalGeneration, BartTokenizer

# ── Configuration ────────────────────────────────────────────────────────────
DATA_PATH = "folio-train.jsonl"
MODEL_NAME = "facebook/bart-large"

BOTTLENECK_SIZE = 8
MAX_LENGTH = 613
BATCH_SIZE = 3          # 16 exhausts GPU memory at this sequence length
LEARNING_RATE = 1e-5
NUM_EPOCHS = 50
SEED = 2024

CHECKPOINT_PATH = (
    f"bart_ae_ep{NUM_EPOCHS}_lr{LEARNING_RATE:g}_bs{BATCH_SIZE}"
    f"_bottleneck{BOTTLENECK_SIZE}_maxlen{MAX_LENGTH}.pth"
)


class BartAutoencoder(torch.nn.Module):
    """BART encoder and decoder joined by a linear bottleneck.

    The encoder's hidden states are projected down to `bottleneck_size`
    dimensions and back up to the model dimension before decoding, so the
    bottleneck width sets how much information can cross between the two.
    """

    def __init__(self, model_name=MODEL_NAME, bottleneck_size=BOTTLENECK_SIZE):
        super().__init__()
        self.encoder = BartForConditionalGeneration.from_pretrained(model_name)
        self.decoder = BartForConditionalGeneration.from_pretrained(model_name)
        d_model = self.encoder.config.d_model
        self.bottleneck_down = torch.nn.Linear(d_model, bottleneck_size)
        self.bottleneck_up = torch.nn.Linear(bottleneck_size, d_model)

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None):
        encoder_outputs = self.encoder.model.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        compressed = self.bottleneck_down(encoder_outputs.last_hidden_state)
        decompressed = self.bottleneck_up(compressed)
        decoder_outputs = self.decoder(
            inputs_embeds=decompressed, decoder_input_ids=decoder_input_ids
        )
        return decoder_outputs.logits


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_premises(path):
    """Return each example's FOL premises joined into a single string.

    The FOLIO records also carry `conclusion`, `premises` and `label`; only the
    FOL premises are used here, since the baseline reconstructs logical form.
    """
    df = pd.read_json(path, lines=True)
    return [", ".join(sentences) for sentences in df["premises-FOL"]]


def build_dataloader(premises, tokenizer):
    """Tokenize the premises and wrap them in a DataLoader.

    Tensors stay on CPU; batches are moved to the device individually, which
    keeps the full tokenized corpus out of GPU memory.
    """
    inputs = tokenizer(
        premises,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )
    decoder_input_ids = inputs.input_ids.clone()
    dataset = TensorDataset(inputs.input_ids, inputs.attention_mask, decoder_input_ids)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)


def train(model, dataloader, tokenizer, device):
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.train()

    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        sample_reconstruction = None

        for step, batch in enumerate(dataloader):
            input_ids, attention_mask, decoder_input_ids = [
                tensor.to(device) for tensor in batch
            ]

            logits = model(
                input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
            )

            # Shift so that position t predicts token t+1.
            shifted_logits = logits[:, :-1].contiguous().view(-1, logits.size(-1))
            targets = decoder_input_ids[:, 1:].contiguous().view(-1)

            loss = loss_fn(shifted_logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Decode the first batch only, as a readable progress check.
            if step == 0:
                predictions = torch.argmax(logits, dim=-1)
                sample_reconstruction = [
                    tokenizer.decode(pred, skip_special_tokens=True)
                    for pred in predictions
                ]

        mean_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}  mean loss: {mean_loss:.4f}")
        if sample_reconstruction:
            print(f"  sample: {sample_reconstruction[0]}")


def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    premises = load_premises(DATA_PATH)
    print(f"Loaded {len(premises)} examples from {DATA_PATH}")

    tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
    dataloader = build_dataloader(premises, tokenizer)

    model = BartAutoencoder().to(device)
    train(model, dataloader, tokenizer, device)

    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"Saved weights to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
