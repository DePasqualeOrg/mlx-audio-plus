# Chatterbox Multilingual Support Analysis

**Date:** December 15, 2025
**Source:** https://huggingface.co/ResembleAI/chatterbox
**PyTorch Reference:** https://github.com/resemble-ai/chatterbox

## Executive Summary

**Good news: The existing MLX port already supports the multilingual architecture.** The T3 model, S3Gen decoder, and VoiceEncoder are architecturally identical between English and multilingual variants. The only differences are:

1. **Text vocabulary size**: 704 (English) vs 2454 (Multilingual)
2. **Text tokenizer**: `EnTokenizer` vs `MTLTokenizer` with language-specific preprocessing
3. **Weight file**: `t3_cfg.safetensors` vs `t3_mtl23ls_v2.safetensors`

Adding multilingual support requires **moderate effort**, primarily around tokenizer changes and conversion script updates.

---

## 1. Model Files Comparison

### English Model Files
```
ResembleAI/chatterbox/
├── t3_cfg.safetensors       # 2.1 GB - English T3 (704 vocab)
├── s3gen.safetensors        # 1.0 GB - Identical
├── ve.safetensors           # 5.7 MB - Identical
├── tokenizer.json           # English BPE tokenizer
└── conds.pt                 # Pre-computed conditionals
```

### Multilingual Model Files
```
ResembleAI/chatterbox/
├── t3_mtl23ls_v2.safetensors  # 2.1 GB - Multilingual T3 (2454 vocab)
├── t3_23lang.safetensors      # 2.1 GB - Alternative multilingual weights
├── s3gen.safetensors          # 1.0 GB - Identical to English
├── ve.safetensors             # 5.7 MB - Identical to English
├── grapheme_mtl_merged_expanded_v1.json  # Multilingual tokenizer vocab
├── mtl_tokenizer.json         # Multilingual tokenizer config
├── Cangjie5_TC.json           # Chinese character encoding map
└── conds.pt                   # Pre-computed conditionals
```

---

## 2. Architecture Comparison

### T3 Model Weights

| Weight Key | English Shape | Multilingual Shape | Notes |
|------------|--------------|-------------------|-------|
| `text_emb.weight` | [704, 1024] | [2352, 1024] | Vocabulary embedding |
| `text_head.weight` | [704, 1024] | [2352, 1024] | Output projection |
| All other weights | Identical | Identical | Same LLaMA backbone |

**Note:** The multilingual model in `t3_23lang.safetensors` uses 2352 tokens (slightly different from the config's 2454), possibly due to vocabulary pruning.

### S3Gen and VoiceEncoder

Both are **100% identical** between English and multilingual:
- Same S3Gen architecture (flow matching decoder + HiFi-GAN)
- Same VoiceEncoder (3-layer LSTM)
- Same S3Tokenizer (speech codec)

---

## 3. Supported Languages

The multilingual model supports **23 languages**:

| Code | Language | Code | Language | Code | Language |
|------|----------|------|----------|------|----------|
| ar | Arabic | he | Hebrew | pl | Polish |
| da | Danish | hi | Hindi | pt | Portuguese |
| de | German | it | Italian | ru | Russian |
| el | Greek | ja | Japanese | sv | Swedish |
| en | English | ko | Korean | sw | Swahili |
| es | Spanish | ms | Malay | tr | Turkish |
| fi | Finnish | nl | Dutch | zh | Chinese |
| fr | French | no | Norwegian | | |

---

## 4. Current MLX Port Status

### Already Implemented

1. **T3Config.multilingual()** - Creates config with `text_tokens_dict_size=2454`
2. **is_multilingual property** - Detects multilingual config
3. **T3 model architecture** - Supports variable text vocabulary size
4. **All other components** - S3Gen, VoiceEncoder, S3Tokenizer work as-is

### Missing for Multilingual

1. **MTLTokenizer class** - Language-specific text preprocessing
2. **Language-specific processors** - Japanese, Chinese, Hebrew, Korean, Russian
3. **Conversion script update** - Download multilingual weights
4. **Generate method update** - Add `language_id` parameter
5. **Punctuation normalization** - Add CJK sentence enders

---

## 5. Required Changes

### 5.1 MTLTokenizer Class (New File)

**File: `mlx_audio/tts/models/chatterbox/tokenizer.py`**

Add a new `MTLTokenizer` class alongside the existing `EnTokenizer`:

```python
# Supported languages
SUPPORTED_LANGUAGES = {
    "ar": "Arabic", "da": "Danish", "de": "German", "el": "Greek",
    "en": "English", "es": "Spanish", "fi": "Finnish", "fr": "French",
    "he": "Hebrew", "hi": "Hindi", "it": "Italian", "ja": "Japanese",
    "ko": "Korean", "ms": "Malay", "nl": "Dutch", "no": "Norwegian",
    "pl": "Polish", "pt": "Portuguese", "ru": "Russian", "sv": "Swedish",
    "sw": "Swahili", "tr": "Turkish", "zh": "Chinese",
}

class MTLTokenizer:
    """Multilingual text tokenizer for Chatterbox TTS."""

    def __init__(self, vocab_file_path: str, model_dir: str = None):
        self.tokenizer = Tokenizer.from_file(vocab_file_path)
        self.cangjie_converter = ChineseCangjieConverter(model_dir)
        self._check_vocab()

    def preprocess_text(self, text: str, language_id: str = None) -> str:
        """Apply language-specific text preprocessing."""
        text = text.lower()
        text = unicodedata.normalize("NFKD", text)

        if language_id == 'zh':
            text = self.cangjie_converter(text)
        elif language_id == 'ja':
            text = hiragana_normalize(text)
        elif language_id == 'he':
            text = add_hebrew_diacritics(text)
        elif language_id == 'ko':
            text = korean_normalize(text)
        elif language_id == 'ru':
            text = add_russian_stress(text)

        return text

    def text_to_tokens(self, text: str, language_id: str = None) -> mx.array:
        """Convert text to token IDs with language tag."""
        text = self.preprocess_text(text, language_id)

        # Prepend language token
        if language_id:
            text = f"[{language_id.lower()}]{text}"

        text = text.replace(" ", SPACE)
        token_ids = self.tokenizer.encode(text).ids
        return mx.array([token_ids], dtype=mx.int32)
```

### 5.2 Language-Specific Preprocessors

These are **optional dependencies** that enhance quality for specific languages:

| Language | Processor | Python Package | Purpose |
|----------|-----------|----------------|---------|
| Japanese | `hiragana_normalize()` | `pykakasi` | Convert kanji to hiragana |
| Chinese | `ChineseCangjieConverter` | `spacy_pkuseg` | Convert hanzi to Cangjie codes |
| Hebrew | `add_hebrew_diacritics()` | `dicta_onnx` | Add nikud (vowel marks) |
| Korean | `korean_normalize()` | Built-in | Decompose syllables to Jamo |
| Russian | `add_russian_stress()` | `russian_text_stresser` | Add stress marks |

**Fallback behavior:** If the optional package is not installed, the preprocessor returns the text unchanged with a warning.

### 5.3 Conversion Script Update

**File: `mlx_audio/tts/models/chatterbox/scripts/convert.py`**

Add `--multilingual` flag:

```python
def download_chatterbox_weights(cache_dir: Path, multilingual: bool = False) -> Path:
    """Download Chatterbox weights from Hugging Face."""
    from huggingface_hub import snapshot_download

    # Select appropriate files based on model type
    if multilingual:
        allow_patterns = [
            "ve.safetensors",
            "t3_mtl23ls_v2.safetensors",  # Multilingual T3
            "s3gen.safetensors",
            "grapheme_mtl_merged_expanded_v1.json",  # Multilingual tokenizer
            "Cangjie5_TC.json",  # Chinese encoding
        ]
    else:
        allow_patterns = [
            "ve.safetensors",
            "t3_cfg.safetensors",  # English T3
            "s3gen.safetensors",
            "tokenizer.json",  # English tokenizer
        ]

    ckpt_dir = Path(
        snapshot_download(
            repo_id="ResembleAI/chatterbox",
            allow_patterns=allow_patterns,
            cache_dir=cache_dir,
        )
    )
    return ckpt_dir


def convert_all(..., multilingual: bool = False):
    # ...

    # Convert T3 with appropriate weights
    t3_file = "t3_mtl23ls_v2.safetensors" if multilingual else "t3_cfg.safetensors"
    t3_weights = load_pytorch_safetensors(ckpt_dir / t3_file)

    # Use multilingual config
    from mlx_audio.tts.models.chatterbox.config import T3Config
    t3_config = T3Config.multilingual() if multilingual else T3Config.english_only()
    t3 = T3(t3_config)
    # ...

    # Copy appropriate tokenizer files
    if multilingual:
        shutil.copy(ckpt_dir / "grapheme_mtl_merged_expanded_v1.json",
                    output_dir / "tokenizer.json")
        shutil.copy(ckpt_dir / "Cangjie5_TC.json",
                    output_dir / "Cangjie5_TC.json")
    else:
        shutil.copy(ckpt_dir / "tokenizer.json", output_dir / "tokenizer.json")

    # Update config.json
    config = {
        "model_type": "chatterbox",
        "multilingual": multilingual,
        "t3_config": {
            "text_tokens_dict_size": 2454 if multilingual else 704,
        }
    }
```

### 5.4 Model Class Updates

**File: `mlx_audio/tts/models/chatterbox/chatterbox.py`**

Update `_init_tokenizers()` to handle multilingual:

```python
def _init_tokenizers(self, model_path: Path) -> None:
    """Initialize text tokenizer from model path."""
    # Check for multilingual tokenizer first
    mtl_tokenizer_path = model_path / "grapheme_mtl_merged_expanded_v1.json"
    en_tokenizer_path = model_path / "tokenizer.json"

    if mtl_tokenizer_path.exists():
        from .tokenizer import MTLTokenizer
        self.tokenizer = MTLTokenizer(mtl_tokenizer_path, model_path)
        self._is_multilingual = True
    elif en_tokenizer_path.exists():
        from .tokenizer import EnTokenizer
        self.tokenizer = EnTokenizer(en_tokenizer_path)
        self._is_multilingual = False
```

Update `generate()` to accept `language_id`:

```python
def generate(
    self,
    text: str,
    language_id: str = None,  # NEW
    audio_prompt: Optional[mx.array] = None,
    # ... other params
) -> Generator[GenerationResult, None, None]:
    """
    Generate speech from text.

    Args:
        text: Input text to synthesize
        language_id: Language code for multilingual models (e.g., "en", "fr", "zh")
                     Ignored for English-only models.
    """
    # Validate language_id for multilingual models
    if self._is_multilingual and language_id:
        if language_id.lower() not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language: {language_id}")

    # Tokenize with language ID
    if hasattr(self.tokenizer, 'text_to_tokens'):
        if self._is_multilingual:
            text_tokens = self.tokenizer.text_to_tokens(text, language_id=language_id)
        else:
            text_tokens = self.tokenizer.text_to_tokens(text)
```

### 5.5 Punctuation Normalization Update

**File: `mlx_audio/tts/models/chatterbox/chatterbox.py`**

Update `punc_norm()` to handle CJK punctuation:

```python
def punc_norm(text: str) -> str:
    # ... existing code ...

    # Add full stop if no ending punc (include CJK punctuation)
    sentence_enders = {".", "!", "?", "-", ",", "、", "，", "。", "？", "！"}
    if not any(text.endswith(p) for p in sentence_enders):
        text += "."

    return text
```

---

## 6. Implementation Checklist

### Phase 1: Core Tokenizer Support

- [ ] Add `MTLTokenizer` class to `tokenizer.py`
- [ ] Add `SUPPORTED_LANGUAGES` constant
- [ ] Implement `ChineseCangjieConverter` class
- [ ] Implement language preprocessing functions (with fallbacks)
- [ ] Update `punc_norm()` for CJK punctuation

### Phase 2: Model Integration

- [ ] Update `_init_tokenizers()` to detect multilingual
- [ ] Add `language_id` parameter to `generate()`
- [ ] Add `_is_multilingual` flag to Model class
- [ ] Update config loading to handle multilingual T3Config

### Phase 3: Conversion Script

- [ ] Add `--multilingual` flag to `convert.py`
- [ ] Update `download_chatterbox_weights()` for multilingual files
- [ ] Update `convert_all()` to use correct T3 weights
- [ ] Copy multilingual tokenizer files
- [ ] Update config.json with multilingual flag

### Phase 4: Testing

- [ ] Test English model still works
- [ ] Test multilingual model with various languages
- [ ] Test fallback behavior when language packages not installed
- [ ] Verify output quality matches PyTorch reference

---

## 7. Usage Examples

### Converting Multilingual Model

```bash
# Convert multilingual model
python -m mlx_audio.tts.models.chatterbox.scripts.convert \
    --multilingual \
    --output-dir ./Chatterbox-Multilingual-fp16

# Convert with quantization
python -m mlx_audio.tts.models.chatterbox.scripts.convert \
    --multilingual \
    --quantize \
    --output-dir ./Chatterbox-Multilingual-4bit
```

### Using Multilingual Model

```python
from mlx_audio.tts.models.chatterbox import Model

# Load multilingual model
model = Model.from_pretrained("mlx-community/Chatterbox-Multilingual-fp16")

# Generate French speech
french_text = "Bonjour, comment ça va?"
wav = model.generate(
    french_text,
    language_id="fr",
    audio_prompt=ref_audio,
    audio_prompt_sr=24000
)

# Generate Chinese speech
chinese_text = "你好，今天天气真不错。"
wav = model.generate(
    chinese_text,
    language_id="zh",
    audio_prompt=ref_audio,
    audio_prompt_sr=24000
)
```

### Command Line

```bash
# French
mlx_audio.tts --model mlx-community/Chatterbox-Multilingual-fp16 \
    --text "Bonjour, comment ça va?" \
    --language fr \
    --ref_audio reference.wav

# Japanese
mlx_audio.tts --model mlx-community/Chatterbox-Multilingual-fp16 \
    --text "こんにちは、お元気ですか？" \
    --language ja \
    --ref_audio reference.wav
```

---

## 8. Optional Dependencies

For best quality in specific languages, install optional packages:

```bash
# Japanese (kanji to hiragana conversion)
pip install pykakasi

# Chinese (word segmentation for Cangjie encoding)
pip install spacy-pkuseg

# Hebrew (diacritics/nikud)
pip install dicta-onnx

# Russian (stress marks)
pip install russian-text-stresser
```

Without these packages, the model will still work but may have reduced quality for those languages.

---

## 9. References

- [Chatterbox HuggingFace](https://huggingface.co/ResembleAI/chatterbox)
- [Chatterbox GitHub](https://github.com/resemble-ai/chatterbox)
- [MTLTokenizer Source](https://github.com/resemble-ai/chatterbox/blob/main/src/chatterbox/models/tokenizers/tokenizer.py)
- [Multilingual TTS Source](https://github.com/resemble-ai/chatterbox/blob/main/src/chatterbox/mtl_tts.py)

---

## 10. Summary

| Aspect | Difficulty | Notes |
|--------|------------|-------|
| T3 Architecture | Already done | Config already supports 2454 vocab |
| S3Gen / VoiceEncoder | Already done | Identical to English |
| MTLTokenizer | Medium | New class with language preprocessing |
| Language processors | Low | Optional, graceful fallback |
| Conversion script | Low | Add --multilingual flag |
| Generate method | Low | Add language_id parameter |

**Total effort: Moderate** - The hardest work (porting the T3/S3Gen architecture) is already complete. The remaining work is primarily tokenizer-related.
