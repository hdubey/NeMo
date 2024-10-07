# script for verifying full-duplex data loader changes

from nemo.collections.multimodal.speech_llm.data.lhotse_dataset import LhotseAudioQuestionAnswerDataset
from lhotse import CutSet

# dummy CutSet or load a correct cutset
dummy_cuts = CutSet.from_jsonl('path_to_your_cutset.jsonl')

# Initialize the dataset
dataset = LhotseAudioQuestionAnswerDataset(
    text_processor=my_text_processor,  # Pass your text processor
    default_context="What can I help you with?",
    tokens_to_generate=50,
    pad_to_max_length=True,
    max_seq_length=512
)

# Load a batch
data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
batch = next(iter(data_loader))

print(batch)
print(batch.keys()) 
print(batch['user_query_audio'].shape)

