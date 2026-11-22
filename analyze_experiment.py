"""Аудит A/B-теста. Запуск: python analyze_experiment.py [путь_к_CSV]."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import chisquare, chi2_contingency, ttest_ind, ks_2samp, fisher_exact
from statsmodels.stats.proportion import proportions_ztest, confint_proportions_2indep
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf
import statsmodels.api as sm

ALPHA = 0.05
HERE = Path(__file__).resolve().parent
CSV_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / 'exp_data_seed_1849273.csv'


def two_props(df, field='converted'):
    """test-control risk difference, относительный lift, z-test и Newcombe CI."""
    a = df.loc[df.variant.eq('control'), field].astype(int)
    b = df.loc[df.variant.eq('test'), field].astype(int)
    if len(a) == 0 or len(b) == 0:
        return None
    pc, pt = a.mean(), b.mean()
    z, p = proportions_ztest([b.sum(), a.sum()], [len(b), len(a)])
    ci = confint_proportions_2indep(int(b.sum()), len(b), int(a.sum()), len(a),
                                   compare='diff', method='newcomb', alpha=ALPHA)
    return dict(n_control=len(a), n_test=len(b), control=pc, test=pt,
                diff=pt-pc, lift=pt/pc-1 if pc else np.nan,
                ci_low=ci[0], ci_high=ci[1], z=z, p=p)


def adjusted_standardization(df):
    """Стандартизация по совместным стратам weekend × platform.

    Оценивает тест-контроль в общей (пуловой) структуре страт. SE через
    биномиальную дисперсию и независимые наблюдения; при пустых стратах
    выдаёт ошибку, а не подменяет оценку.
    """
    weights = df.groupby(['weekend', 'platform']).size().div(len(df))
    estimates = []
    variance = 0.0
    for key, weight in weights.items():
        part = df[(df.weekend == key[0]) & (df.platform == key[1])]
        c = part[part.variant == 'control'].converted
        t = part[part.variant == 'test'].converted
        if min(len(c), len(t)) == 0:
            raise ValueError(f'Нет перекрытия групп в страте {key}')
        estimates.append(weight * (t.mean() - c.mean()))
        variance += weight**2 * (t.mean()*(1-t.mean())/len(t) +
                                  c.mean()*(1-c.mean())/len(c))
    rd = sum(estimates)
    se = np.sqrt(variance)
    return rd, (rd-1.96*se, rd+1.96*se), se


def main():
    df = pd.read_csv(CSV_PATH)
    required = {'user_id','variant','converted','error_flag','weekend','platform','entry_time'}
    assert required.issubset(df.columns), 'Отсутствуют обязательные столбцы'
    print('=== PRE-LAUNCH: структура и качество данных ===')
    print('Размер:', df.shape, '; пропуски:', df.isna().sum().to_dict())
    print('Дубликаты user_id:', df.user_id.duplicated().sum(),
          '; дубликаты строк:', df.duplicated().sum())
    for field in ('converted','error_flag','weekend'):
        assert df[field].isin([0,1]).all(), f'Недопустимые значения: {field}'
    assert df.variant.isin(['control','test']).all()
    assert df.platform.isin([0,1,2]).all()
    assert df.entry_time.between(0,1).all()
    assert df.user_id.is_unique
    print('entry_time: доля суток [0, 1); пример:',
          pd.to_timedelta(df.entry_time.head(3), unit='D').astype(str).tolist())
    print('\n=== Базовые метрики ===')
    print(df.groupby('variant').agg(n=('user_id','size'),
                                    conversion=('converted','mean'),
                                    errors=('error_flag','mean'),
                                    weekend=('weekend','mean'),
                                    entry_time=('entry_time','mean')).to_string())

    count = df.variant.value_counts().reindex(['control','test'])
    srm_stat, srm_p = chisquare(count.to_numpy(), f_exp=np.repeat(len(df)/2, 2))
    print(f'\nSRM: chi2={srm_stat:.4f}, p={srm_p:.6g}, '
          f'вердикт={"SRM обнаружен" if srm_p<ALPHA else "SRM нет"}')

    xtab = pd.crosstab(df.variant, df.weekend)
    weekend_chi2, weekend_p, _, _ = chi2_contingency(xtab)
    dt = df[df.variant=='test'].entry_time
    dc = df[df.variant=='control'].entry_time
    time_t = ttest_ind(dt, dc, equal_var=False)
    time_ks = ks_2samp(dt, dc)
    print(f'weekend (chi2): p={weekend_p:.6g}, '
          f'тест/контроль частота={df.groupby("variant").weekend.mean().to_dict()}')
    print(f'entry_time (Welch): p={time_t.pvalue:.6g}; '
          f'KS: p={time_ks.pvalue:.6g}; '
          f'средние: test={dt.mean():.5f}, control={dc.mean():.5f}')

    print('\n=== ПЕРВИЧНАЯ МЕТРИКА (пока без решения о rollout) ===')
    conv = two_props(df)
    print('control={control:.4%}, test={test:.4%}, абсолютный эффект={diff:+.4%}, '
          'относительный lift={lift:+.2%}, 95% CI RD=[{ci_low:+.4%}; {ci_high:+.4%}], '
          'p={p:.6g}'.format(**conv))

    print('\n=== GUARDRAIL: error_flag ===')
    guard = two_props(df, 'error_flag')
    print('control={control:.4%}, test={test:.4%}, RD={diff:+.4%}, '
          '95% CI=[{ci_low:+.4%}; {ci_high:+.4%}], p(two-sided)={p:.6g}'.format(**guard))
    # Односторонний точный критерий: test > control.
    tab = pd.crosstab(df.variant, df.error_flag).reindex(index=['test','control'],
                                                         columns=[1,0], fill_value=0)
    odds, guard_one_p = fisher_exact(tab.to_numpy(), alternative='greater')
    print(f'Fisher one-sided H1(test>control): p={guard_one_p:.6g}, odds={odds:.3f}')

    print('\n=== СЕГМЕНТЫ platform: 3 теста с коррекцией Holm ===')
    segments = []
    for platform in sorted(df.platform.unique()):
        res = two_props(df[df.platform==platform])
        res['platform'] = platform
        segments.append(res)
    seg = pd.DataFrame(segments)
    seg['p_holm'] = multipletests(seg.p, alpha=ALPHA, method='holm')[1]
    print(seg[['platform','n_control','n_test','control','test','diff','lift',
               'ci_low','ci_high','p','p_holm']].to_string(index=False))
    print('Примечание: коррекция Holm относится только к трём сегментным тестам; '
          'исследовательский выбор дополнительных срезов тоже нужно учитывать.')

    print('\n=== ENTRY_TIME: диагностируем возможный time bias ===')
    d = df.copy()
    d['time_bin'] = pd.cut(d.entry_time, bins=np.linspace(0,1,11), include_lowest=True)
    print(pd.crosstab(d.time_bin, d.variant, normalize='index').to_string())
    logistic = smf.glm('is_test ~ entry_time', data=d.assign(is_test=(d.variant=='test').astype(int)),
                       family=sm.families.Binomial()).fit()
    print(f'Логистическая связь variant~entry_time: beta={logistic.params["entry_time"]:.4f}, '
          f'p={logistic.pvalues["entry_time"]:.6g}. '
          'Даже значимая связь не доказывает факт peeking (нет логов остановки).')

    print('\n=== КОРРЕКЦИЯ ПО НАБЛЮДАЕМЫМ КОВАРИАТАМ ===')
    adj, adj_ci, adj_se = adjusted_standardization(df)
    print(f'Сырая RD={conv["diff"]:+.4%}; стандартизированная RD={adj:+.4%}, '
          f'приближённый 95% CI=[{adj_ci[0]:+.4%}; {adj_ci[1]:+.4%}]')
    reg = smf.glm('converted ~ C(variant) + weekend + C(platform)',
                  data=df, family=sm.families.Binomial()).fit(cov_type='HC3')
    v = 'C(variant)[T.test]'
    print(f'Регрессия (логистическая, HC3): OR={np.exp(reg.params[v]):.3f}, '
          f'95% CI OR={tuple(np.exp(reg.conf_int().loc[v]).round(3))}, '
          f'p={reg.pvalues[v]:.6g}. OR не равен lift конверсии.')
    print('Стандартизация не исправляет SRM, нарушенную рандомизацию, '
          'искусственный эффект в platform=2 или guardrail.')

    blockers = []
    if srm_p < ALPHA: blockers.append('SRM')
    if weekend_p < ALPHA: blockers.append('дисбаланс weekend')
    if time_t.pvalue < ALPHA or time_ks.pvalue < ALPHA: blockers.append('time bias (не доказательство peeking)')
    if guard_one_p < ALPHA and guard['diff'] > 0: blockers.append('ухудшение guardrail')
    print('\n=== РЕШЕНИЕ ===')
    print('НЕ ROLLOUT. Повторить с изменениями. Основания:', ', '.join(blockers) or 'нет блокеров')
    print('Нужны 50/50 независимое хеш-назначение, AA/SRM-check, заранее заданный '
          'срок/объём без peeking, семейство гипотез и поправки, guardrail stop-rule, '
          'проверка платформенных событий и повторный запуск.')
    return dict(df=df, conv=conv, guard=guard, seg=seg, srm_p=srm_p,
                weekend_p=weekend_p, time_p=time_t.pvalue, time_ks_p=time_ks.pvalue,
                guard_one_p=guard_one_p, adj=adj, adj_ci=adj_ci)


if __name__ == '__main__':
    main()
