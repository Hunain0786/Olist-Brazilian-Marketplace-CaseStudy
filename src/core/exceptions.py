


class PredictorError(Exception):
    pass


class ModelNotLoadedError(PredictorError):
    pass


class ArtifactLoadError(PredictorError):

    def __init__(self, artifact: str, cause: Exception):
        self.artifact = artifact
        self.cause = cause
        super().__init__(f"Failed to load artifact '{artifact}': {cause}")


class FeatureEngineeringError(PredictorError):

    def __init__(self, detail: str):
        super().__init__(f"Feature engineering error: {detail}")


class InvalidInputError(PredictorError):

    def __init__(self, field: str, detail: str):
        self.field = field
        super().__init__(f"Invalid value for '{field}': {detail}")


class PredictionInternalError(PredictorError):

    def __init__(self, detail: str):
        super().__init__(f"Internal prediction error: {detail}")
