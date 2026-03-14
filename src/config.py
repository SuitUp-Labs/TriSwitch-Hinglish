from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
    random_seed: int = 42
    selected_metrics: tuple[str, ...] = ("bleu", "bertscore")
    required_fields: tuple[str, ...] = (
        "id",
        "base",
        "variant_topic_fronting",
        "variant_emphasis_shift",
        "tokens_hindi",
        "tokens_english",
        "switch_points",
        "domain",
        "pattern",
    )

    @property
    def data_raw_dir(self) -> Path:
        return self.project_root / "data" / "raw"

    @property
    def data_interim_dir(self) -> Path:
        return self.project_root / "data" / "interim"

    @property
    def data_processed_dir(self) -> Path:
        return self.project_root / "data" / "processed"

    @property
    def outputs_logs_dir(self) -> Path:
        return self.project_root / "outputs" / "logs"

    @property
    def raw_dataset_path(self) -> Path:
        return self.data_raw_dir / "triswitch_hinglish_500.json"

    @property
    def cleaned_dataset_path(self) -> Path:
        return self.data_interim_dir / "cleaned_dataset.csv"

    @property
    def validated_dataset_path(self) -> Path:
        return self.data_interim_dir / "validated_dataset.csv"

    @property
    def pairwise_dataset_path(self) -> Path:
        return self.data_interim_dir / "pairwise_dataset.csv"

    @property
    def pairwise_features_path(self) -> Path:
        return self.data_interim_dir / "pairwise_dataset_with_features.csv"

    @property
    def human_validation_path(self) -> Path:
        return self.data_processed_dir / "human_validation.csv"

    @property
    def metric_scores_path(self) -> Path:
        return self.data_processed_dir / "metric_scores.csv"

    @property
    def metric_summary_path(self) -> Path:
        return self.data_processed_dir / "metric_summary.csv"

    @property
    def dataset_sanity_log_path(self) -> Path:
        return self.outputs_logs_dir / "dataset_sanity.txt"


CONFIG = PipelineConfig()
