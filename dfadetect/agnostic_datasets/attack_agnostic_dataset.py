import pandas as pd


from dfadetect.agnostic_datasets.asvspoof_dataset import ASVSpoofDataset
from dfadetect.agnostic_datasets.base_dataset import SimpleAudioFakeDataset
from dfadetect.agnostic_datasets.fakeavceleb_dataset import FakeAVCelebDataset
from dfadetect.agnostic_datasets.wavefake_dataset import WaveFakeDataset



class AttackAgnosticDataset(SimpleAudioFakeDataset):

    def __init__(
        self,
        asvspoof_path=None,
        wavefake_path=None,
        fakeavceleb_path=None,
        fold_num=0,
        fold_subset="val",
        transform=None,
        oversample=True,
        undersample=False,
        return_label=True,
        reduced_number=None,
    ):
        super().__init__(fold_num, fold_subset, transform, return_label)

        datasets = []

        # Only load dataset if path is provided and valid
        if asvspoof_path is not None and asvspoof_path != "" and str(asvspoof_path).lower() != "none":
            try:
                asvspoof_dataset = ASVSpoofDataset(asvspoof_path, fold_num=fold_num, fold_subset=fold_subset)
                datasets.append(asvspoof_dataset)
                print(f"✓ Loaded ASVspoof dataset")
            except Exception as e:
                print(f"⚠ Skipping ASVspoof: {e}")

        if wavefake_path is not None and wavefake_path != "" and str(wavefake_path).lower() != "none":
            try:
                wavefake_dataset = WaveFakeDataset(wavefake_path, fold_num=fold_num, fold_subset=fold_subset)
                datasets.append(wavefake_dataset)
                print(f"✓ Loaded WaveFake dataset")
            except Exception as e:
                print(f"⚠ Skipping WaveFake: {e}")

        if fakeavceleb_path is not None and fakeavceleb_path != "" and str(fakeavceleb_path).lower() != "none":
            try:
                fakeavceleb_dataset = FakeAVCelebDataset(fakeavceleb_path, fold_num=fold_num, fold_subset=fold_subset)
                datasets.append(fakeavceleb_dataset)
                print(f"✓ Loaded FakeAVCeleb dataset")
            except Exception as e:
                print(f"⚠ Skipping FakeAVCeleb: {e}")

        # Check if at least one dataset was loaded
        if len(datasets) == 0:
            raise ValueError("❌ No datasets loaded! Provide at least one valid dataset path.")

        self.samples = pd.concat(
            [ds.samples for ds in datasets],
            ignore_index=True
        )

        if oversample:
            self.oversample_dataset()
        elif undersample:
            self.undersample_dataset()

        if reduced_number is not None:
            self.samples = self.samples.sample(reduced_number, replace=True, random_state=42)

    def oversample_dataset(self):
        # Check what labels we actually have
        unique_labels = self.samples['label'].unique()
        print(f"Unique labels in dataset: {unique_labels}")
        
        samples = self.samples.groupby(by=['label'])
        
        # Handle both string and tuple groupby keys
        group_keys = list(samples.groups.keys())
        
        # Find bonafide and spoof samples
        bonafide_key = None
        spoof_key = None
        
        for key in group_keys:
            # Handle tuple keys from groupby
            label = key[0] if isinstance(key, tuple) else key
            if label == "bonafide":
                bonafide_key = key
            elif label == "spoof":
                spoof_key = key
        
        if bonafide_key is None or spoof_key is None:
            print(f"Warning: Could not find bonafide/spoof labels. Available keys: {group_keys}")
            print("Skipping oversampling.")
            return
        
        bona_length = len(samples.groups[bonafide_key])
        spoof_length = len(samples.groups[spoof_key])
        
        print(f"Bonafide samples: {bona_length}, Spoof samples: {spoof_length}")

        diff_length = spoof_length - bona_length

        if diff_length < 0:
            raise NotImplementedError("Bonafide oversampling not implemented")

        if diff_length > 0:
            bonafide = samples.get_group(bonafide_key).sample(diff_length, replace=True)
            self.samples = pd.concat([self.samples, bonafide], ignore_index=True)
            print(f"Oversampled bonafide by {diff_length} samples. Total samples now: {len(self.samples)}")

    def undersample_dataset(self):
        samples = self.samples.groupby(by=['label'])
        
        # Handle both string and tuple groupby keys
        group_keys = list(samples.groups.keys())
        
        bonafide_key = None
        spoof_key = None
        
        for key in group_keys:
            label = key[0] if isinstance(key, tuple) else key
            if label == "bonafide":
                bonafide_key = key
            elif label == "spoof":
                spoof_key = key
        
        if bonafide_key is None or spoof_key is None:
            print(f"Warning: Could not find bonafide/spoof labels. Skipping undersampling.")
            return
        
        bona_length = len(samples.groups[bonafide_key])
        spoof_length = len(samples.groups[spoof_key])

        if spoof_length < bona_length:
            raise NotImplementedError("Bonafide undersampling not implemented")

        if spoof_length > bona_length:
            spoofs = samples.get_group(spoof_key).sample(bona_length, replace=True)
            self.samples = pd.concat([samples.get_group(bonafide_key), spoofs], ignore_index=True)

    def get_bonafide_only(self):
        samples = self.samples.groupby(by=['label'])
        
        # Handle both string and tuple groupby keys
        group_keys = list(samples.groups.keys())
        bonafide_key = None
        
        for key in group_keys:
            label = key[0] if isinstance(key, tuple) else key
            if label == "bonafide":
                bonafide_key = key
                break
        
        if bonafide_key is None:
            raise ValueError(f"No bonafide samples found. Available labels: {group_keys}")
        
        self.samples = samples.get_group(bonafide_key)
        return self.samples

    def get_spoof_only(self):
        samples = self.samples.groupby(by=['label'])
        
        # Handle both string and tuple groupby keys
        group_keys = list(samples.groups.keys())
        spoof_key = None
        
        for key in group_keys:
            label = key[0] if isinstance(key, tuple) else key
            if label == "spoof":
                spoof_key = key
                break
        
        if spoof_key is None:
            raise ValueError(f"No spoof samples found. Available labels: {group_keys}")
        
        self.samples = samples.get_group(spoof_key)
        return self.samples
