# AnuLM API: real requests and responses

Recorded 2026-10-18 00:18 against `http://127.0.0.1:8000` by `run_demo.py`.
Every block below is what the server actually returned; ids, timings and sampled
text change from run to run. Long base64 strings are shortened.

## 01. List every model and which parts are borrowed

`GET /v1/models` → **200** in 0.001 s

```bash
curl http://127.0.0.1:8000/v1/models
```

Response:

```json
{
  "object": "list",
  "data": [
    {
      "id": "anulm-chat",
      "from_scratch": true,
      "endpoints": [
        "/v1/chat/completions",
        "/v1/completions"
      ]
    },
    {
      "id": "anulm-answer",
      "from_scratch": false,
      "endpoints": [
        "/v1/chat/completions",
        "/v1/answer"
      ]
    },
    {
      "id": "anulm-vision",
      "from_scratch": false,
      "endpoints": [
        "/v1/chat/completions"
      ]
    },
    {
      "id": "anulm-asr",
      "from_scratch": false,
      "endpoints": [
        "/v1/audio/transcriptions"
      ]
    },
    {
      "id": "system-tts",
      "from_scratch": false,
      "endpoints": [
        "/v1/audio/speech"
      ]
    },
    {
      "id": "anulm-decide-banking",
      "from_scratch": true,
      "endpoints": [
        "/v1/decide"
      ]
    },
    {
      "id": "anulm-decide-hindi",
      "from_scratch": true,
      "endpoints": [
        "/v1/decide"
      ]
    },
    {
      "id": "anulm-decide-general",
      "from_scratch": true,
      "endpoints": [
        "/v1/decide"
      ]
    },
    {
      "id": "modernbert-decide-banking",
      "from_scratch": false,
      "endpoints": [
        "/v1/decide"
      ]
    },
    {
      "id": "modernbert-decide-hindi",
      "from_scratch": false,
      "endpoints": [
        "/v1/decide"
      ]
    }
  ]
}
```

## 02. Chat (English)

`POST /v1/chat/completions` → **200** in 5.704 s

```bash
curl http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "anulm-chat", "messages": [{"role": "user", "content": "Give me three tips for saving money every month."}], "max_tokens": 60, "temperature": 0.7, "repetition_penalty": 1.3, "seed": 7}'
```

Request:

```json
{
  "model": "anulm-chat",
  "messages": [
    {
      "role": "user",
      "content": "Give me three tips for saving money every month."
    }
  ],
  "max_tokens": 60,
  "temperature": 0.7,
  "repetition_penalty": 1.3,
  "seed": 7
}
```

Response:

```json
{
  "id": "chatcmpl-120a7a60325e4c0d834815b1",
  "object": "chat.completion",
  "created": 1792262824,
  "model": "anulm-chat",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Here are three tips.\n1. Set a goal and prioritize your goals\n2. Make your budget manageable\n3. Don't spend too much money on your home and the things you use\n4. Use clear, specific language to describe your actions\n5. Practice while you are traveling\n6. Consider yourself ready to spend your money"
      },
      "finish_reason": "length"
    }
  ],
  "usage": {
    "prompt_tokens": 16,
    "completion_tokens": 60,
    "total_tokens": 76
  }
}
```

## 03. Chat (Hindi)

`POST /v1/chat/completions` → **200** in 2.201 s

```bash
curl http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "anulm-chat", "messages": [{"role": "user", "content": "सुबह जल्दी उठने के तीन फ़ायदे बताइए।"}], "max_tokens": 100, "temperature": 0.7, "repetition_penalty": 1.3, "seed": 7}'
```

Request:

```json
{
  "model": "anulm-chat",
  "messages": [
    {
      "role": "user",
      "content": "सुबह जल्दी उठने के तीन फ़ायदे बताइए।"
    }
  ],
  "max_tokens": 100,
  "temperature": 0.7,
  "repetition_penalty": 1.3,
  "seed": 7
}
```

Response:

```json
{
  "id": "chatcmpl-f0c5eaf6179643728a9760f0",
  "object": "chat.completion",
  "created": 1792262827,
  "model": "anulm-chat",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "शाम को जल्दी उठना एक प्रकार का खेल है। इसे करना या जाना बहुत ही सामान्य बात है लेकिन कुछ लोग इसका मजाक उड़ा देते हैं क्योंकि वह अपने साथ हुए दुविधाओं और थके हुए काम की वजह से किसी कारण से नहीं सोते हैं।"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 17,
    "completion_tokens": 53,
    "total_tokens": 70
  }
}
```

## 04. Chat, streamed (server-sent events)

`POST /v1/chat/completions` → **200** in 1.877 s

```bash
curl http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "anulm-chat", "messages": [{"role": "user", "content": "Write one sentence about the monsoon."}], "max_tokens": 48, "temperature": 0.7, "repetition_penalty": 1.3, "seed": 7, "stream": true, "stream_options": {"include_usage": true}}'
```

Request:

```json
{
  "model": "anulm-chat",
  "messages": [
    {
      "role": "user",
      "content": "Write one sentence about the monsoon."
    }
  ],
  "max_tokens": 48,
  "temperature": 0.7,
  "repetition_penalty": 1.3,
  "seed": 7,
  "stream": true,
  "stream_options": {
    "include_usage": true
  }
}
```

Response:

```json
{
  "assembled_text": " My question is how many days have passed since I was born, and since my birth so far, I'm in a state of monsoon drought. The day of being born is usually the last day after which the weather is warm enough",
  "events": [
    {
      "id": "chatcmpl-20eaffbd9043496ca40d131d",
      "object": "chat.completion.chunk",
      "created": 1792262829,
      "model": "anulm-chat",
      "choices": [
        {
          "index": 0,
          "delta": {
            "role": "assistant",
            "content": ""
          },
          "finish_reason": null
        }
      ]
    },
    {
      "id": "chatcmpl-20eaffbd9043496ca40d131d",
      "object": "chat.completion.chunk",
      "created": 1792262829,
      "model": "anulm-chat",
      "choices": [
        {
          "index": 0,
          "delta": {
            "content": " My"
          },
          "finish_reason": null
        }
      ]
    },
    "... 48 more chunks ...",
    {
      "id": "chatcmpl-20eaffbd9043496ca40d131d",
      "object": "chat.completion.chunk",
      "created": 1792262829,
      "model": "anulm-chat",
      "choices": [],
      "usage": {
        "prompt_tokens": 15,
        "completion_tokens": 48,
        "total_tokens": 63
      }
    },
    "[DONE]"
  ]
}
```

## 05. Raw text completion, cut at the first full stop

`POST /v1/completions` → **200** in 0.302 s

```bash
curl http://127.0.0.1:8000/v1/completions -H "Content-Type: application/json" \
  -d '{"model": "anulm-chat", "prompt": "The three primary colours are", "max_tokens": 32, "temperature": 0, "stop": ["."]}'
```

Request:

```json
{
  "model": "anulm-chat",
  "prompt": "The three primary colours are",
  "max_tokens": 32,
  "temperature": 0,
  "stop": [
    "."
  ]
}
```

Response:

```json
{
  "id": "cmpl-7dcdcc784fec4740a290c104",
  "object": "text_completion",
  "created": 1792262831,
  "model": "anulm-chat",
  "choices": [
    {
      "index": 0,
      "text": " red, yellow, and blue",
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 5,
    "completion_tokens": 7,
    "total_tokens": 12
  }
}
```

## 06. Wikipedia lookup with sources (/v1/answer)

`POST /v1/answer` → **200** in 15.467 s

```bash
curl http://127.0.0.1:8000/v1/answer -H "Content-Type: application/json" \
  -d '{"question": "Who built the Taj Mahal?"}'
```

Request:

```json
{
  "question": "Who built the Taj Mahal?"
}
```

Response:

```json
{
  "object": "answer",
  "question": "Who built the Taj Mahal?",
  "answer": "Shah Jahan",
  "found": true,
  "sources": [
    "Taj Mahal",
    "Taj Mahal",
    "Taj Mahal"
  ],
  "from": "wikipedia",
  "seconds": 11.98
}
```

## 07. The same lookup through the OpenAI chat endpoint (anulm-answer)

`POST /v1/chat/completions` → **200** in 0.321 s

```bash
curl http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "anulm-answer", "messages": [{"role": "user", "content": "भारत की राजधानी क्या है?"}]}'
```

Request:

```json
{
  "model": "anulm-answer",
  "messages": [
    {
      "role": "user",
      "content": "भारत की राजधानी क्या है?"
    }
  ]
}
```

Response:

```json
{
  "id": "chatcmpl-a453701a9d8946518df94eee",
  "object": "chat.completion",
  "created": 1792262847,
  "model": "anulm-answer",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "नई दिल्ली"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0
  },
  "sources": [
    "दिल्ली",
    "भारत की राजधानियों की सूची",
    "नई दिल्ली"
  ]
}
```

## 08. Decide: route banking messages (a batch of three)

`POST /v1/decide` → **200** in 1.467 s

```bash
curl http://127.0.0.1:8000/v1/decide -H "Content-Type: application/json" \
  -d '{"model": "anulm-decide-banking", "input": ["I still haven'\''t received my new card", "My card was stolen this morning", "hi, quick question"], "threshold": 0.7, "top": 3}'
```

Request:

```json
{
  "model": "anulm-decide-banking",
  "input": [
    "I still haven't received my new card",
    "My card was stolen this morning",
    "hi, quick question"
  ],
  "threshold": 0.7,
  "top": 3
}
```

Response:

```json
{
  "object": "decision",
  "model": "anulm-decide-banking",
  "from_scratch": true,
  "results": [
    {
      "input": "I still haven't received my new card",
      "choice": "card_arrival",
      "probabilities": {
        "card_arrival": 0.9221,
        "get_physical_card": 0.014,
        "getting_virtual_card": 0.0105
      },
      "confidence": 0.9221,
      "action": "auto-route"
    },
    {
      "input": "My card was stolen this morning",
      "choice": "lost_or_stolen_card",
      "probabilities": {
        "lost_or_stolen_card": 0.8627,
        "lost_or_stolen_phone": 0.1162,
        "compromised_card": 0.0144
      },
      "confidence": 0.8627,
      "action": "auto-route"
    },
    {
      "input": "hi, quick question",
      "choice": "extra_charge_on_statement",
      "probabilities": {
        "extra_charge_on_statement": 0.1758,
        "get_physical_card": 0.0963,
        "unable_to_verify_identity": 0.0857
      },
      "confidence": 0.1758,
      "action": "send to human"
    }
  ],
  "seconds": 0.956
}
```

## 09. Decide: Hindi assistant intents

`POST /v1/decide` → **200** in 0.842 s

```bash
curl http://127.0.0.1:8000/v1/decide -H "Content-Type: application/json" \
  -d '{"model": "anulm-decide-hindi", "input": "कल सुबह छह बजे का अलार्म लगा दो", "top": 3}'
```

Request:

```json
{
  "model": "anulm-decide-hindi",
  "input": "कल सुबह छह बजे का अलार्म लगा दो",
  "top": 3
}
```

Response:

```json
{
  "object": "decision",
  "model": "anulm-decide-hindi",
  "from_scratch": true,
  "results": [
    {
      "input": "कल सुबह छह बजे का अलार्म लगा दो",
      "choice": "alarm_set",
      "probabilities": {
        "alarm_set": 0.9491,
        "alarm_remove": 0.0438,
        "alarm_query": 0.0064
      },
      "confidence": 0.9491,
      "action": "auto-route"
    }
  ],
  "seconds": 0.256
}
```

## 10. Decide: zero-shot with your own options

`POST /v1/decide` → **200** in 3.559 s

```bash
curl http://127.0.0.1:8000/v1/decide -H "Content-Type: application/json" \
  -d '{"model": "anulm-decide-general", "input": ["The delivery came two weeks late and the box was crushed", "Thank you, the staff were wonderful", "Do you ship to Canada?"], "options": ["complaint about a delivery", "praise for the service", "question about a product", "request to place an order"], "threshold": 0.7, "top": 2}'
```

Request:

```json
{
  "model": "anulm-decide-general",
  "input": [
    "The delivery came two weeks late and the box was crushed",
    "Thank you, the staff were wonderful",
    "Do you ship to Canada?"
  ],
  "options": [
    "complaint about a delivery",
    "praise for the service",
    "question about a product",
    "request to place an order"
  ],
  "threshold": 0.7,
  "top": 2
}
```

Response:

```json
{
  "object": "decision",
  "model": "anulm-decide-general",
  "from_scratch": true,
  "results": [
    {
      "input": "The delivery came two weeks late and the box was crushed",
      "choice": "complaint about a delivery",
      "probabilities": {
        "complaint about a delivery": 0.9507,
        "praise for the service": 0.0312
      },
      "confidence": 0.9507,
      "action": "auto-route"
    },
    {
      "input": "Thank you, the staff were wonderful",
      "choice": "praise for the service",
      "probabilities": {
        "praise for the service": 0.9786,
        "question about a product": 0.0108
      },
      "confidence": 0.9786,
      "action": "auto-route"
    },
    {
      "input": "Do you ship to Canada?",
      "choice": "praise for the service",
      "probabilities": {
        "praise for the service": 0.6605,
        "request to place an order": 0.1805
      },
      "confidence": 0.6605,
      "action": "send to human"
    }
  ],
  "seconds": 0.384
}
```

## 11. Vision: caption an image (image only, no text)

`POST /v1/chat/completions` → **200** in 6.56 s

```bash
# replace "@data/..." with a data URL: data:image/jpeg;base64,$(base64 -w0 data/astronaut.jpg)
curl http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "anulm-vision", "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "@data/astronaut.jpg"}}]}]}'
```

Request:

```json
{
  "model": "anulm-vision",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "image_url",
          "image_url": {
            "url": "@data/astronaut.jpg"
          }
        }
      ]
    }
  ]
}
```

Response:

```json
{
  "id": "chatcmpl-d36056e500a549dca35f4693",
  "object": "chat.completion",
  "created": 1792262859,
  "model": "anulm-vision",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "A woman in a orange jumpsuit is smiling and smiling."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0
  }
}
```

## 12. Vision: ask a question about the image (short answer)

`POST /v1/chat/completions` → **200** in 5.045 s

```bash
# replace "@data/..." with a data URL: data:image/jpeg;base64,$(base64 -w0 data/astronaut.jpg)
curl http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "anulm-vision", "messages": [{"role": "user", "content": [{"type": "text", "text": "What color is the flag?"}, {"type": "image_url", "image_url": {"url": "@data/astronaut.jpg"}}]}]}'
```

Request:

```json
{
  "model": "anulm-vision",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "What color is the flag?"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "@data/astronaut.jpg"
          }
        }
      ]
    }
  ]
}
```

Response:

```json
{
  "id": "chatcmpl-6e0acda4acfa4fefb541b586",
  "object": "chat.completion",
  "created": 1792262864,
  "model": "anulm-vision",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "red"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0
  }
}
```

## 13. Text to speech (returns WAV)

`POST /v1/audio/speech` → **200** in 2.248 s

```bash
curl http://127.0.0.1:8000/v1/audio/speech -H "Content-Type: application/json" \
  -d '{"model": "system-tts", "input": "My card was stolen this morning, please block it.", "response_format": "wav"}'
```

Request:

```json
{
  "model": "system-tts",
  "input": "My card was stolen this morning, please block it.",
  "response_format": "wav"
}
```

Response:

```json
{
  "saved_to": "responses/13_text_to_speech.wav",
  "bytes": 117804
}
```

## 14. Speech to text (multipart upload of the WAV from step 13)

`POST /v1/audio/transcriptions` → **200** in 5.671 s

```bash
curl http://127.0.0.1:8000/v1/audio/transcriptions -F file=@responses/13_text_to_speech.wav -F model=anulm-asr -F language=en -F response_format=verbose_json
```

Response:

```json
{
  "task": "transcribe",
  "language": "en",
  "duration": 2.67,
  "text": "my card was strollin this morning please block it",
  "segments": []
}
```

## 15. Error: unknown model (OpenAI-style error body)

`POST /v1/chat/completions` → **404** in 0.002 s

```bash
curl http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "gpt-4o", "messages": [{"role": "user", "content": "hello"}]}'
```

Request:

```json
{
  "model": "gpt-4o",
  "messages": [
    {
      "role": "user",
      "content": "hello"
    }
  ]
}
```

Response:

```json
{
  "error": {
    "message": "Model 'gpt-4o' is not served by this endpoint; use one of: anulm-chat, anulm-answer, anulm-vision",
    "type": "invalid_request_error",
    "param": "model",
    "code": "model_not_found"
  }
}
```

## 16. Error: anulm-decide-general without options

`POST /v1/decide` → **400** in 0.001 s

```bash
curl http://127.0.0.1:8000/v1/decide -H "Content-Type: application/json" \
  -d '{"model": "anulm-decide-general", "input": "where is my parcel"}'
```

Request:

```json
{
  "model": "anulm-decide-general",
  "input": "where is my parcel"
}
```

Response:

```json
{
  "error": {
    "message": "anulm-decide-general needs an options list.",
    "type": "invalid_request_error",
    "param": "options",
    "code": null
  }
}
```

## Sample data scored through /v1/decide

### `anulm-decide-banking` on `data/banking_queries.csv`: 100% of 16 (16/16 auto-routed at 100%)

| query | expected | choice | confidence | action |
|---|---|---|---|---|
| I still haven't received my new card | card_arrival | card_arrival | 0.92 | auto-route |
| My card was stolen this morning | lost_or_stolen_card | lost_or_stolen_card | 0.86 | auto-route |
| What exchange rate do you use for euros? | exchange_rate | exchange_rate | 0.99 | auto-route |
| I sent money to my brother three days ago and he still hasn't got it | transfer_not_received_by_recipient | transfer_not_received_by_recipient | 0.94 | auto-route |
| I typed the wrong PIN too many times and now it's blocked | pin_blocked | pin_blocked | 1.00 | auto-route |
| My top-up didn't go through | top_up_failed | top_up_failed | 0.78 | auto-route |
| Why was I charged a fee for taking cash out of the ATM? | cash_withdrawal_charge | cash_withdrawal_charge | 1.00 | auto-route |
| The shop said my card was declined | declined_card_payment | declined_card_payment | 0.83 | auto-route |
| How do I change my PIN? | change_pin | change_pin | 1.00 | auto-route |
| How do I activate the card you sent me? | activate_my_card | activate_my_card | 0.97 | auto-route |
| I want my money back for this purchase | request_refund | request_refund | 0.95 | auto-route |
| The ATM gave me less cash than I asked for | wrong_amount_of_cash_received | wrong_amount_of_cash_received | 1.00 | auto-route |
| Can I add my card to Apple Pay? | apple_pay_or_google_pay | apple_pay_or_google_pay | 1.00 | auto-route |
| There's a direct debit on my statement I don't recognise | direct_debit_payment_not_recognised | direct_debit_payment_not_recognised | 1.00 | auto-route |
| I think someone else has been using my card details | compromised_card | compromised_card | 0.97 | auto-route |
| How old do you have to be to open an account? | age_limit | age_limit | 1.00 | auto-route |

### `anulm-decide-hindi` on `data/hindi_queries.csv`: 100% of 8 (8/8 auto-routed at 100%)

| query | expected | choice | confidence | action |
|---|---|---|---|---|
| कल सुबह छह बजे का अलार्म लगा दो | alarm_set | alarm_set | 0.95 | auto-route |
| आज दिल्ली में मौसम कैसा रहेगा | weather_query | weather_query | 1.00 | auto-route |
| कोई पुराना हिंदी गाना चलाओ | play_music | play_music | 0.96 | auto-route |
| अभी कितने बजे हैं | datetime_query | datetime_query | 0.89 | auto-route |
| कमरे की बत्ती बंद कर दो | iot_hue_lightoff | iot_hue_lightoff | 0.84 | auto-route |
| आज की ताज़ा ख़बरें सुनाओ | news_query | news_query | 1.00 | auto-route |
| पनीर बटर मसाला कैसे बनाते हैं | cooking_recipe | cooking_recipe | 0.99 | auto-route |
| मेरे लिए कल मुंबई की ट्रेन का टिकट बुक करो | transport_ticket | transport_ticket | 0.99 | auto-route |
