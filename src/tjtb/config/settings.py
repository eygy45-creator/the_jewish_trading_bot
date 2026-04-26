"""Top-level application settings."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict

from tjtb.config.backtest_settings import BacktestSettings
from tjtb.config.feature_settings import FeatureSettings
from tjtb.config.instrument_specs import InstrumentSpec, MNQ_DEFAULT_SPEC
from tjtb.config.model_settings import ModelSettings
from tjtb.config.risk_settings import RiskSettings
from tjtb.config.session_research_settings import SessionResearchSettings


class AppSettings(BaseSettings):
    """Environment-overridable app settings."""

    model_config = SettingsConfigDict(env_prefix="TJTB_", env_nested_delimiter="__")

    instrument: InstrumentSpec = MNQ_DEFAULT_SPEC
    features: FeatureSettings = FeatureSettings()
    model: ModelSettings = ModelSettings()
    backtest: BacktestSettings = BacktestSettings()
    risk: RiskSettings = RiskSettings()
    session_research: SessionResearchSettings = SessionResearchSettings()
    log_level: str = "INFO"
