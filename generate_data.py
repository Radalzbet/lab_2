"""Генерация уникального датасета; seed фиксируется для воспроизводимости."""
import numpy as np
import pandas as pd
from pathlib import Path

# Сгенерирован однократно; измените только если создаёте новый независимый датасет.
MY_SEED = 1849273
np.random.seed(MY_SEED)

# Единственное изменение диапазона исходника: больше наблюдений для устойчивости
# анализа редкой guardrail-метрики. Механизмы нарушений сохранены.
n_users = np.random.randint(8500, 10000, size=1)[0]
true_effect = 0
expected_ratio = 0.5

time_of_entry = np.random.uniform(0, 1, n_users)
variant = np.where(time_of_entry > 0.4, 'test', 'control')  # затем перезаписывается!
p_ratio = np.random.uniform(low=0.4001, high=0.599999, size=1)[0]
variant = np.random.choice(['control', 'test'], size=n_users, p=[p_ratio, 1-p_ratio])
weekend = np.random.binomial(1, 0.3, n_users)
p_assign = np.random.uniform(low=0.40001, high=0.499999, size=1)[0]
variant = np.where(
    (variant == 'test') & (weekend == 1), 'test',
    np.where(
        (variant == 'control') & (weekend == 0), 'control',
        np.random.choice(['control', 'test'], size=n_users, p=[p_assign, 1-p_assign])
    )
)
conversion_prob = 0.1 + 0.05 * weekend
converted = np.random.binomial(1, conversion_prob, n_users)
p_platform2 = np.random.uniform(low=0.02, high=0.15, size=1)[0]
p_platform1 = np.random.uniform(low=0.25, high=0.35, size=1)[0]
platform = np.random.choice([0, 1, 2], size=n_users,
                            p=[1-p_platform1-p_platform2, p_platform1, p_platform2])
converted = np.where((platform == 2) & (variant == 'test'),
                     np.random.binomial(1, 0.2, size=len(converted)), converted)
error_flag = np.zeros(n_users, dtype=int)
error_flag[variant == 'control'] = np.random.binomial(1, 0.02, sum(variant == 'control'))
error_flag[variant == 'test'] = np.random.binomial(1, 0.06, sum(variant == 'test'))
df = pd.DataFrame({
    'user_id': range(n_users), 'variant': variant, 'converted': converted,
    'error_flag': error_flag, 'weekend': weekend, 'platform': platform,
    'entry_time': time_of_entry
})
output = Path(__file__).resolve().parent / f'exp_data_seed_{MY_SEED}.csv'
df.to_csv(output, index=False)
print(f'Создан файл {output.name}, N={len(df)}, seed={MY_SEED}')
