# Полный аудит A/B-эксперимента — seed 1849273

## Содержимое

- `exp_data_seed_1849273.csv` — уникальный датасет (9127 строк).
- `generate_data.py` — воспроизводимый генератор, seed зафиксирован. Исходная логика искажения сохранена; диапазон `n_users` изменён на 8500–9999 для статистической мощности.
- `experiment_protocol_audit.ipynb` — выполненный Jupyter Notebook: диагностика, 5 графиков, 10 исполняемых ячеек, статтесты, текстовые выводы.
- `analyze_experiment.py` — независимый анализ теми же методами из командной строки.
- `results.txt` — фактический протокол вычислений.

## Запуск

```bash
pip install numpy pandas scipy statsmodels matplotlib seaborn notebook nbclient ipykernel
python analyze_experiment.py
jupyter notebook experiment_protocol_audit.ipynb
```

Открывайте ноутбук из папки, в которой размещён CSV. Для Google Colab нужно загрузить `.ipynb` и `.csv` в рабочую среду Colab, затем выполнить ячейки.

**Не запускайте генератор заново при сдаче уже зафиксированного датасета**: он воспроизводит тот же seed и может перезаписать CSV идентичными строками.

## Главные результаты

| Критерий | Результат |
|---|---|
| N, control/test | 9127; 4726/4401 |
| SRM | χ²=11.573, p=0.000669 |
| weekend control/test | 9.99% / 51.69%, p≈0 |
| raw conversion control/test | 11.02% / 14.02%, p=1.50e-5 |
| absolute RD conversion | +2.995 п.п., 95% CI +1.638…+4.359 п.п. |
| error rate control/test | 2.18% / 5.66%, one-sided Fisher p=2.20e-18 |
| platform 2, after Holm | p=1.22e-7, artificial segment artifact |
| entry_time group effect | Welch p=0.981, KS p=0.862 — нет свидетельства временного перекоса |
| standardized conversion RD | +0.798 п.п., 95% CI −0.782…+2.379 п.п. |
| Verdict | **НЕ ROLLOUT. Исправить протокол и повторить тест.** |

Важно: симуляция в исходном генераторе первоначально создаёт временную зависимость назначения, но затем **перезаписывает** её. Поэтому `entry_time` не доказывает peeking, а раннюю остановку принципиально нельзя проверить без логов. Корректировка по `weekend × platform` не исправляет SRM, guardrail и запись искусственного эффекта в platform=2.

## Публикация

Загрузите CSV, `.ipynb`, `.py` и README в репозиторий GitHub/GitVerse; на образовательную платформу отправьте только ссылку, если это предписано преподавателем. Этот архив **не опубликован** в удалённом репозитории автоматически.
