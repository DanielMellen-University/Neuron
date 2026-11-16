# Fixed M4 perplexity text

[wikitext2-slice.txt](wikitext2-slice.txt) is a 1,599-byte prefix of the
WikiText-2 raw test split. Its SHA-256 is
`5c4d7185e322a055c34ef0ba1c0511973f916adc637b4159ee86a1be8300da4a`.
It was fixed before collecting any M4 quality results.

Source: [Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext),
revision `b08601e04326c79dfdd32d625aee71d232d685c3`, configuration
`wikitext-2-raw-v1`, test split. The [source Parquet file](https://huggingface.co/datasets/Salesforce/wikitext/resolve/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-2-raw-v1/test-00000-of-00001.parquet)
is 732,610 bytes, SHA-256
`5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91`.
WikiText was assembled by Stephen Merity, Caiming Xiong, James Bradbury and
Richard Socher from Wikipedia articles. The excerpt begins with the
[Robert Boulter](https://en.wikipedia.org/wiki/Robert_Boulter) article.
The dataset card supplies the
[Creative Commons Attribution-ShareAlike 3.0 license](https://creativecommons.org/licenses/by-sa/3.0/).
That license applies to this text excerpt; the repository's MIT license
does not replace it. The only changes are taking a prefix and adding a
final newline.

To regenerate, install PyArrow outside the repository, then run from the
repository root. Evaluation itself uses only Python's standard library
and the committed text, with no download or PyArrow requirement.

```sh
python3 -m pip install --target /tmp/neuron-m4-python pyarrow
PYTHONPATH=/tmp/neuron-m4-python python3 - <<'PY'
import hashlib
from pathlib import Path
import urllib.request
import pyarrow.parquet as pq
url = ('https://huggingface.co/datasets/Salesforce/wikitext/resolve/'
       'b08601e04326c79dfdd32d625aee71d232d685c3/'
       'wikitext-2-raw-v1/test-00000-of-00001.parquet')
source = Path('/tmp/neuron-m4-wikitext.parquet')
urllib.request.urlretrieve(url, source)
assert hashlib.sha256(source.read_bytes()).hexdigest() == '5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91'
text = ''.join(pq.read_table(source)['text'].to_pylist()).encode()[:1600]
text = text[:text.rfind(b' ')] + b'\n'
assert hashlib.sha256(text).hexdigest() == '5c4d7185e322a055c34ef0ba1c0511973f916adc637b4159ee86a1be8300da4a'
Path('bench/wikitext2-slice.txt').write_bytes(text)
PY
```
