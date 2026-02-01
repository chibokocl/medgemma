# medgemma

Medical AI on Apple Silicon – MedGemma 1.5 4B via MLX.

## Install

```bash
pip install medgemma
```

## Usage

```python
from medgemma import MedGemma

mg = MedGemma()
response = mg.ask("What are symptoms of diabetes?")
print(response.text)
```

### CLI

```bash
medgemma ask "What are symptoms of diabetes?"
medgemma ask "Describe this X-ray" --image xray.png
medgemma setup
medgemma info
```
