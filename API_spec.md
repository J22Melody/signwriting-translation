# API Spec

All paramters are optional.

`POST /api/translate/spoken2sign`

requests

```
{
    "country_code": "us",
    "language_code": "en",
    "text": "hello",
    "translation_type": "dict"
}
```

returns

```
{
    "country_code": "us",
    "direction": "spoken2sign",
    "language_code": "en",
    "n_best": "3",
    "text": "hello",
    "translation_type": "dict",
    "translations": [
        "M518x518S30007482x483S15a11485x482S26500515x461",
        "M518x518S14c20481x453S27106503x483",
        "M518x518S30a00482x483S33e00482x483"
    ]
}
```

`POST /api/translate/sign2spoken`

requests

```
{
    "country_code": "us",
    "language_code": "en",
    "text": "M528x518S15a07472x487S1f010490x503S26507515x483 M517x515S1e020494x485S21801482x492 S38800464x496       L532x594S2e74c475x569S30d00482x482S2e700517x555S14220500x520S14228473x533 L518x571S2ff00482x482S30d00482x482S22a03480x536S15a51461x548S15a00494x508 L536x527S10e41465x485S2892a507x494S10e4a474x512S2892a506x472 S38700463x496 L529x568S15a37477x442S18250488x453S28802507x431S15a37481x536S10051492x547S26901470x522 L526x599S15a50474x528S15a31479x576S24103494x545S30122482x476 S38900464x493 M530x679S1001a490x568S10041487x541S2e718466x654S22f04492x527S14220498x607S2e73c514x642S14228467x618S30a00482x482 M547x531S1853f473x504S18537513x497S2b711474x479S2b700508x469S2b700528x480S2b711452x492 S38800464x496",
    "translation_type": "sent",
    "n_best": 5
}
```

returns

```
{
    "country_code": "us",
    "direction": "sign2spoken",
    "language_code": "en",
    "n_best": 5,
    "text": "M528x518S15a07472x487S1f010490x503S26507515x483 M517x515S1e020494x485S21801482x492 S38800464x496 L532x594S2e74c475x569S30d00482x482S2e700517x555S14220500x520S14228473x533 L518x571S2ff00482x482S30d00482x482S22a03480x536S15a51461x548S15a00494x508 L536x527S10e41465x485S2892a507x494S10e4a474x512S2892a506x472 S38700463x496 L529x568S15a37477x442S18250488x453S28802507x431S15a37481x536S10051492x547S26901470x522 L526x599S15a50474x528S15a31479x576S24103494x545S30122482x476 S38900464x493 M530x679S1001a490x568S10041487x541S2e718466x654S22f04492x527S14220498x607S2e73c514x642S14228467x618S30a00482x482 M547x531S1853f473x504S18537513x497S2b711474x479S2b700508x469S2b700528x480S2b711452x492 S38800464x496",
    "translation_type": "sent",
    "translations": [
        "Verse 21. The wicked borrow and again, and the godly gives them.",
        "Verse 21. The wicked borrow and never repay, but the godly gives them.",
        "Verse 21. The wicked borrow and never repay, but the godly gives it.",
        "Verse 21. The wicked borrow and never repay, but the godly gives them.",
        "Verse 21. The wicked borrow, never repay, but the godly gives them."
    ]
}
```



