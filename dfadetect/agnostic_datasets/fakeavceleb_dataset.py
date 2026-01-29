from pathlib import Path

import pandas as pd

from dfadetect.agnostic_datasets.base_dataset import SimpleAudioFakeDataset


FAKEAVCELEB_KFOLD_SPLIT = {
    0: {
        "train": ['rtvc', 'faceswap-wav2lip'],
        "test": ['fsgan-wav2lip'],
        "val": ['wav2lip'],
        "bonafide_partition": [0.7, 0.15],
        "seed": 42
    },
    1: {
        "train": ['fsgan-wav2lip', 'wav2lip'],
        "test": ['rtvc'],
        "val": ['faceswap-wav2lip'],
        "bonafide_partition": [0.7, 0.15],
        "seed": 43
    },
    2: {
        "train": ['faceswap-wav2lip', 'fsgan-wav2lip'],
        "test": ['wav2lip'],
        "val": ['rtvc'],
        "bonafide_partition": [0.7, 0.15],
        "seed": 44
    }
}


class FakeAVCelebDataset(SimpleAudioFakeDataset):

    audio_folder = "FakeAVCeleb-audio"
    audio_extension = ".flac"
    metadata_file = Path(audio_folder) / "meta_data.csv"
    subsets = ("train", "dev", "eval")

    def __init__(self, path, fold_num=0, fold_subset="train", transform=None):
        super().__init__(fold_num, fold_subset, transform)
        self.path = path

        self.fold_num, self.fold_subset = fold_num, fold_subset
        self.allowed_attacks = FAKEAVCELEB_KFOLD_SPLIT[fold_num][fold_subset]
        self.bona_partition = FAKEAVCELEB_KFOLD_SPLIT[fold_num]["bonafide_partition"]
        self.seed = FAKEAVCELEB_KFOLD_SPLIT[fold_num]["seed"]

        self.metadata = self.get_metadata()

        self.samples = pd.concat([self.get_fake_samples(), self.get_real_samples()], ignore_index=True)

    def get_metadata(self):
        """Generate metadata by scanning audio directories"""
        
        # Try to read existing metadata file first
        metadata_path = Path(self.path) / self.metadata_file
        if metadata_path.exists():
            md = pd.read_csv(metadata_path)
            md["audio_type"] = md["type"].apply(lambda x: x.split("-")[-1])
            return md
        
        # If metadata doesn't exist, generate it from folder structure
        print(f"Metadata file not found. Generating from directory structure...")
        
        records = []
        base_path = Path(self.path)
        
        # Map folder names to method and type
        folder_mapping = {
            'FakeVideo-FakeAudio': {'method': 'unknown', 'type': 'FakeVideo-FakeAudio', 'audio_type': 'FakeAudio'},
            'FakeVideo-RealAudio': {'method': 'real', 'type': 'FakeVideo-RealAudio', 'audio_type': 'RealAudio'},
            'RealVideo-FakeAudio': {'method': 'unknown', 'type': 'RealVideo-FakeAudio', 'audio_type': 'FakeAudio'},
            'RealVideo-RealAudio': {'method': 'real', 'type': 'RealVideo-RealAudio', 'audio_type': 'RealAudio'}
        }
        
        # Scan each folder
        for folder_name, metadata_info in folder_mapping.items():
            folder_path = base_path / folder_name
            
            if not folder_path.exists():
                print(f"Warning: Folder not found: {folder_path}")
                continue
            
            # Find all .flac files
            audio_files = list(folder_path.rglob('*.flac'))
            print(f"Found {len(audio_files)} files in {folder_name}")
            
            for audio_file in audio_files:
                # Extract information from file path
                # Example: FakeVideo-FakeAudio/African/men/id00076/00109_10_id00476_wavtolip.flac
                relative_path = audio_file.relative_to(base_path)
                parts = relative_path.parts
                
                # Try to extract method from filename or use folder default
                filename = audio_file.name
                method = metadata_info['method']
                
                # Check if filename contains known attack methods
                if 'wav2lip' in filename.lower() or 'wavtolip' in filename.lower():
                    method = 'wav2lip'
                elif 'faceswap' in filename.lower():
                    method = 'faceswap-wav2lip'
                elif 'fsgan' in filename.lower():
                    method = 'fsgan-wav2lip'
                elif 'rtvc' in filename.lower():
                    method = 'rtvc'
                
                # Extract source/user_id from path if possible
                source = parts[1] if len(parts) > 1 else 'unknown'
                if len(parts) > 3:
                    source = parts[3]  # This should be the id (e.g., id00076)
                
                records.append({
                    'filename': filename,
                    'path': str(relative_path.parent),
                    'method': method,
                    'type': metadata_info['type'],
                    'audio_type': metadata_info['audio_type'],
                    'source': source
                })
        
        if not records:
            raise ValueError(f"No audio files found in {base_path}")
        
        md = pd.DataFrame(records)
        print(f"Generated metadata: {len(md)} samples")
        print(f"Methods found: {md['method'].value_counts().to_dict()}")
        print(f"Audio types: {md['audio_type'].value_counts().to_dict()}")
        
        return md

    def get_fake_samples(self):
        samples = {
            "user_id": [],
            "sample_name": [],
            "attack_type": [],
            "label": [],
            "path": []
        }

        for attack_name in self.allowed_attacks:
            fake_samples = self.metadata[
                (self.metadata["method"] == attack_name) & (self.metadata["audio_type"] == "FakeAudio")
            ]

            for index, sample in fake_samples.iterrows():
                samples["user_id"].append(sample["source"])
                samples["sample_name"].append(Path(sample["filename"]).stem)
                samples["attack_type"].append(sample["method"])
                samples["label"].append("spoof")
                samples["path"].append(self.get_file_path(sample))

        return pd.DataFrame(samples)

    def get_real_samples(self):
        samples = {
            "user_id": [],
            "sample_name": [],
            "attack_type": [],
            "label": [],
            "path": []
        }

        samples_list = self.metadata[
            (self.metadata["method"] == "real") & (self.metadata["audio_type"] == "RealAudio")
        ]

        samples_list = self.split_real_samples(samples_list)

        for index, sample in samples_list.iterrows():
            samples["user_id"].append(sample["source"])
            samples["sample_name"].append(Path(sample["filename"]).stem)
            samples["attack_type"].append("-")
            samples["label"].append("bonafide")
            samples["path"].append(self.get_file_path(sample))

        return pd.DataFrame(samples)

    def get_file_path(self, sample):
        path = "/".join([self.audio_folder, *sample["path"].split("/")[1:]])
        return Path(self.path) / path / Path(sample["filename"]).with_suffix(self.audio_extension)


if __name__ == "__main__":
    FAKEAVCELEB_DATASET_PATH = ""

    real = 0
    fake = 0
    for subset in ['train', 'test', 'val']:
        dataset = FakeAVCelebDataset(FAKEAVCELEB_DATASET_PATH, fold_num=2, fold_subset=subset)
        dataset.get_real_samples()
        real += len(dataset)

        print('real', len(dataset))

        dataset = FakeAVCelebDataset(FAKEAVCELEB_DATASET_PATH, fold_num=2, fold_subset=subset)
        dataset.get_fake_samples()
        fake += len(dataset)

        print('fake', len(dataset))

    print(real, fake)
