# Sample outputs

Real replies from `tools_sweep.py`, which runs every released checkpoint over thousands of held-out prompts. Selected by the rule in `tools_samples.py` -- nearest the median repetition among replies that stopped on their own -- so these are typical, not the best in the file. The aggregate numbers for the whole sweep are in each section and in `docs/RESULTS.md`.

## Writing Python

Asked in prose, from MBPP and HumanEval. The model writes the whole function.

*1138 prompts, 0 errors, 0 empty, 85.6% stopped on their own, 4.7% repeated 4-grams on average, 1942 ms per reply.*

**mbpp** — prompt:

```
Write a function to sort a given matrix in ascending order according to the sum of its rows.
```

reply:

```
def sort_matrix(matrix):
    # Initialize the sum of rows to 0
    total_rows = 0
    
    # Iterate through each row in the matrix
    for row in matrix:
        # Add the row to the total sum
        total_rows += row
        
        # Increment the row count by 1
        row += 1
    
    # Return the sorted matrix
    return matrix
```

**mbpp** — prompt:

```
Write a python function to find the volume of a triangular prism.
```

reply:

```
def find_triangle_volume(prism):
    volume = 0
    for row in prism:
        for col in row:
            volume += col
    return volume
```

**mbpp** — prompt:

```
Write a function to split a string at lowercase letters.
```

reply:

```
def split_string(s):
    return s.lower().split()
```

**mbpp** — prompt:

```
Write a function to find sequences of lowercase letters joined with an underscore.
```

reply:

```
def find_sequences(sequence):
    sequence = sequence.lower()
    sequences = []
    for char in sequence:
        if char == '_':
            sequences.append(char)
    return sequences
```

**humaneval** — prompt:

```
from typing import List


def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
```

reply:

```
for i in range(len(numbers)):
        for j in range(i+1, len(numbers)):
            if abs(numbers[i] - numbers[j]) < threshold and abs(numbers[j] - numbers[i]) < threshold:
                return True
    return False
```

**humaneval** — prompt:

```
from typing import List


def below_zero(operations: List[int]) -> bool:
    """ You're given a list of deposit and withdrawal operations on a bank account that starts with
    zero balance. Your task is to detect if at any point the balance of account fallls below zero, and
    at that point function should return True. Otherwise it should return False.
    >>> below_zero([1, 2, 3])
    False
    >>> below_zero([1, 2, -4, 5])
    True
    """
```

reply:

```
for operation in operations:
        if operation < 0:
            return True
    return False
```

**humaneval** — prompt:

```
from typing import List


def intersperse(numbers: List[int], delimeter: int) -> List[int]:
    """ Insert a number 'delimeter' between every two consecutive elements of input list `numbers'
    >>> intersperse([], 4)
    []
    >>> intersperse([1, 2, 3], 4)
    [1, 4, 2, 4, 3]
    """
```

reply:

```
result = []
    for i in range(len(numbers) - 1):
        result.append(numbers[i] + numbers[i + 1])
    return result
```

**humaneval** — prompt:

```
from typing import List, Tuple


def sum_product(numbers: List[int]) -> Tuple[int, int]:
    """ For a given list of integers, return a tuple consisting of a sum and a product of all the integers in a list.
    Empty sum should be equal to 0 and empty product should be equal to 1.
    >>> sum_product([])
    (0, 1)
    >>> sum_product([1, 2, 3, 4])
    (10, 24)
    """
```

reply:

```
result = (0, 1)
    for num in numbers:
        result = (result[0] + result[1], result[0] * result[1])
    return result
```

## Translating

FLORES-200 devtest, both directions. The direction is picked from the script.

*2024 prompts, 0 errors, 0 empty, 98.5% stopped on their own, 1.8% repeated 4-grams on average, 1108 ms per reply.*

**flores** — prompt:

```
"We now have 4-month-old mice that are non-diabetic that used to be diabetic," he added.
```

reply:

```
उन्होंने कहा, ‘‘हमारे पास अब 4 महीने के माइनर हैं जो गैर-विविधता का उपयोग करने वाले गैर-विविधता का उपयोग करते हैं।
```

**flores** — prompt:

```
उन्होंने कहा “कि अब हमारे पास 4 महीने उम्र वाले चूहे हैं जिन्हें मधुमेह नहीं है जो मधुमेह के रोगी थे। ”
```

reply:

```
He said that now we have four months of old mice that are not diabetes that were the patients of diabetes.
```

**flores** — prompt:

```
Dr. Ehud Ur, professor of medicine at Dalhousie University in Halifax, Nova Scotia and chair of the clinical and scientific division of the Canadian Diabetes Association cautioned that the research is still in its early days.
```

reply:

```
मिलिफैक्स के दलहौजी विश्वविद्यालय में चिकित्सा के प्रोफेसर डॉ. एहूद उरे ने कोविड-19 के उपचार के लिए लीबिया के डिस्ट्रिक्ट विश्वविद्यालय में चिकित्सा और वैज्ञानिक प्रभाग की अध्यक्षता में इस बात की चेतावनी दी कि अनुसंधान अभी भी अपने शुरुआती दिनों में है।
```

**flores** — prompt:

```
Like some other experts, he is skeptical about whether diabetes can be cured, noting that these findings have no relevance to people who already have Type 1 diabetes.
```

reply:

```
कुछ अन्य विशेषज्ञों की तरह, वह इस बात से अवगत हैं कि क्या डायबिटीज की उपचार की जा सकती है, न कि यह कि इन निष्कर्षों में लोगों को कोई प्रासंगिकता नहीं है जो पहले से ही टाइप 1 मधुमेह के लिए हैं।
```

**flores** — prompt:

```
कुछ अन्य विशेषज्ञों की तरह, उन्हें इस बात पर संदेह है कि क्या मधुमेह को ठीक किया जा सकता है, यह देखते हुए कि इन निष्कर्षों की उन लोगों के लिए कोई प्रासंगिकता नहीं है जिन्हें पहले से ही टाइप 1 मधुमेह है.
```

reply:

```
Like some other experts they suspect that diabetes can be cured, given that there is no relevant relevance for those who have already been type 1 diabetes.
```

**flores** — prompt:

```
सोमवार को स्वीडिश अकादमी में साहित्य के लिए नोबेल समिति के स्थायी सचिव, सारा डेनियस, स्वीडन समिति में सेवरिज रेडियो पर एक रेडियो कार्यक्रम के दौरान सार्वजनिक रूप से घोषणा की, कि साहित्य में 2016 के नोबेल पुरस्कार जीतने को लेकर सीधे बॉब डायलन तक पहुँचने में असमर्थ होकर, अपने प्रयासों को छोड़ चुके थे.
```

reply:

```
The Nobel Committee on Culture at the University of Sweden on Monday announced that the public announcement on the success of the 2016 Nobel Prize, the Seshajani, was not unfulfilled to be the first to win the 2016 Nobel Prize.
```

**flores** — prompt:

```
Danius said, "Right now we are doing nothing. I have called and sent emails to his closest collaborator and received very friendly replies. For now, that is certainly enough."
```

reply:

```
"डैनियस ने कहा, ""आज हम कुछ नहीं कर रहे हैं और मैंने अपने करीबी सहयोगियों को ईमेल भेजा है और बहुत मित्राना जवाब दिया है।"
```

**flores** — prompt:

```
डेनियस ने कहा, “फ़िलहाल हम कुछ नहीं कर रहे हैं. मैंने अपने साथ काम करने वाले लोगों को ईमेल भेजे हैं और उसने उनके फ्रेंडली जवाब भेजे हैं. अभी के लिए, बस इतना ही बताना चाहूँगा.”
```

reply:

```
"""There are no people at the moment, I have sent email to the people who work with me and have sent their friends."
```

## Continuing text

The opening line of a held-out document, in Hindi, English or Python.

*969 prompts, 0 errors, 0 empty, 3.0% stopped on their own, 66.9% repeated 4-grams on average, 2966 ms per reply.*

**python** — prompt:

```
# ramcn/demo2/crowdsourcing/models.py
```

reply:

```
############################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################
```

**python** — prompt:

```
# njvack/ge-mri-rtafni/scanner-console/vendor/dicom/filereader.py
```

reply:

```
############################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################
```

**english** — prompt:

```
http://www.mdservices.nl/en/our-services/loyalty.html
```

reply:

```
Products and related services are available for sale in the United States. The following are the following:
https://www.nl.com/en/our-services/loyalty.html
https://www.nl.com/en/our-services/loyalty.html
https://www.nl.com/en/our-services/loyalty.html
https://www.nl.com/en/our-services/loyalty.html
https://www.nl.com/en/our-services/
```

**hindi** — prompt:

```
विकिपीडिया:पृष्ठ हटाने हेतु चर्चा/फ़ाइलें/Dafan hote dariya 02-2.jpg
```

reply:

```
navbox:Navbox:Navbox:Navbox:Navbox:Navbox:Navbox:Navbox:Navbox:Navbox:Navbox:Navbox:Navbox:Navbox:Navbox:Navbox:Navbox:Navbox:Navbox:Navbox:Navbox:Navbox:Navbox:Navbox:
```

**hindi** — prompt:

```
महाराव शेखा
```

reply:

```
बीबीबीबीबीबीबीबीबीबीबीबीबीबीबीबीबीबीबीबीबीबीबी के रूप में

बीबीबीबीबीबीबीबीबीबीबीबीबीबीबीबीबीबीबीबीबीबीबीबीबीबीबीबीबीबीबीबीबीबीबीबीबीबीबीबीबीबीबीबीबीबीबी के रूप में

बीबीबीबीबीबीबीबीबीबीबीबीबीबीबीबीबीबी
```

**english** — prompt:

```
http://liralustde.cba.pl/q5ne85md681-q.html
```

reply:

```
Q5ne85md691qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq
```

**python** — prompt:

```
# theheros/kbengine/kbe/res/scripts/common/Lib/importlib/test/builtin/test_loader.py
```

reply:

```
############################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################
```

**python** — prompt:

```
# credativ/gofer/src/gofer/agent/config.py
```

reply:

```
############################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################
```
