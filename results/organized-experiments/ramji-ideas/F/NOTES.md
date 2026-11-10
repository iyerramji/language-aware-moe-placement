# F: English/Indic replica position sensitivity

This executes only a placement experiment. It DOES NOT insert, train, remove or test a shared FFN/translator layer. Available marginal expert counts cannot predict altered hidden states or altered router choices.

English/Indic groups come from E, fitted on training counts and held fixed across layers. All variants have96replicas per GPU across48layers and unchanged128logical experts per layer. Early/late windows have12copies per layer; middle32has3; both boundaries have6in each of16layers; uniform has2in all48. The8-layer sliding window is selected using fit data only. No new home migration between layers is assumed.

- G=4: uniform 36.456%, middle-only 32.804% (delta -3.652 points); fit-selected window starts at layer 41, gives 25.556% reduction. These compare equal total copies, not equal copies per layer.

- G=8: uniform 31.107%, middle-only 28.285% (delta -2.822 points); fit-selected window starts at layer 41, gives 22.434% reduction. These compare equal total copies, not equal copies per layer.
