# <img src="https://img.icons8.com/external-flaticons-lineal-color-flat-icons/64/undefined/external-big-data-smart-technology-flaticons-lineal-color-flat-icons-2.png"/> **NLP Research**

# **Relation Extraction in Biomedical using BI-LSTM-CRF model + Bert Embedding + Keras + Tensorflow**

## <img src="https://img.icons8.com/external-smashingstocks-flat-smashing-stocks/64/000000/external-manager-hotel-smashingstocks-flat-smashing-stocks-2.png"/> **`Slamet Riyanto S.Kom., M.M.S.I.`**

## <img src="https://img.icons8.com/external-fauzidea-flat-fauzidea/64/undefined/external-man-avatar-avatar-fauzidea-flat-fauzidea.png"/> **`Dimas Dwi Putra`**

## Architecture
<img src="RE-BILSTM-CRF%20Architecture.png" width="16384">

## Dataset
<img src="RE-DATASET.png" width="4803">

## Visualization
<img src="uji%205/loss_acc.png" width="640">

## Eval
| Entities              | precision | recall | f1-score | support | excecution time | processor | ram  | model | batch size | epochs | length | embedding | Uji |
| --------------------- | --------- | ------ | -------- | ------- | --------------- | --------- | ---- | ----- | ---------- | ------ | ------ | --------- | --- |
| Negative              | 0.73      | 0.75   | 0.74     | 118     | 1.07.48         | cpu       | high | 1     | 16         | 30     | 512    | 128       | 5   |
| Cause Of Disease      | 0.67      | 0.54   | 0.60     | 37      |                 |           |      |       |            |        |        |           |     |
| Treatment Of Diesease | 0.79      | 0.87   | 0.83     | 98      |                 |           |      |       |            |        |        |           |     |
| Association           | 1.00      | 0.25   | 0.40     | 8       |                 |           |      |       |            |        |        |           |     |
| micro avg             |           |        | 0.75     | 261     |                 |           |      |       |            |        |        |           |     |
| macro avg             | 0.80      | 0.60   | 0.64     | 261     |                 |           |      |       |            |        |        |           |     |
| weighted avg          | 0.75      | 0.75   | 0.74     | 261     |                 |           |      |       |            |        |        |           |     |
| F-1 Scores            |           |        | 75%      |         |                 |           |      |       |            |        |        |           |     |

## Predict
```yaml
original: green tea#skin papillomas#Inhibitory effect of green tea on the growth of established skin papillomas in mice.
Predict: Treatment_of_disease
```