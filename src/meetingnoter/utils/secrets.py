import logging
import os

logger = logging.getLogger(__name__)

def _get_secret(key: str) -> str:
    """Helper to securely get credentials from environment or colab userdata."""
    val = os.environ.get(key)
    if val:
        return val

    try:
        from google.colab import userdata
        val = userdata.get(key)
        if val:
            return str(val)
    except Exception as e:
        logger.debug("Colab userdata not available or failed: %s", e)

    msg = f"Missing required credential: {key}"
    raise ValueError(msg)
