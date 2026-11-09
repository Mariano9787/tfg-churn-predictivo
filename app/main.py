"""Four-view read-only TP4 prototype. Run: python -m streamlit run app/main.py"""
from pathlib import Path
import json

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts' / 'final'

st.set_page_config(page_title='Churn | Prototipo TFG', page_icon='📊', layout='wide')
st.title('Prototipo predictivo de abandono de clientes')
st.caption('IBM Telco Customer Churn · escenario de referencia · puntajes fuera de muestra · uso académico')

required = ['portfolio_oof.parquet', 'explanations_oof.parquet', 'holdout_metrics.json',
            'holdout_deciles.csv', 'holdout_calibration.csv', 'global_factors.csv']
missing = [f for f in required if not (ART/f).exists()]
if missing:
    st.error('Primero ejecutá `python -m src.complete_pipeline` desde la raíz del proyecto.')
    st.code('Faltan: ' + ', '.join(missing))
    st.stop()

@st.cache_data
def load():
    return (pd.read_parquet(ART/'portfolio_oof.parquet'),
            pd.read_parquet(ART/'explanations_oof.parquet'),
            json.loads((ART/'holdout_metrics.json').read_text(encoding='utf-8')),
            pd.read_csv(ART/'holdout_deciles.csv'),
            pd.read_csv(ART/'holdout_calibration.csv'),
            pd.read_csv(ART/'global_factors.csv'),
            json.loads((ART/'portfolio_metadata.json').read_text(encoding='utf-8')))

portfolio, explanations, holdout, deciles, calibration, factors, metadata = load()
view = st.sidebar.radio('Vistas', ['Cartera', 'Detalle del cliente', 'Factores de la cartera', 'Desempeño'])
st.sidebar.caption('Los resultados son artefactos precalculados. No se entrena ni puntúa desde la interfaz.')
st.sidebar.caption(f"Modelo seleccionado: {metadata['selected_family']}")

if view == 'Cartera':
    st.subheader('Cartera ordenada por riesgo estimado')
    st.info('Puntajes OOF: cada cliente fue evaluado por una instancia que no se entrenó con ese cliente. No equivalen a una predicción futura validada temporalmente.')
    c1,c2,c3 = st.columns(3)
    c1.metric('Clientes', f'{len(portfolio):,}'.replace(',', '.'))
    c2.metric('Puntaje medio', f"{portfolio['score'].mean():.3f}")
    c3.metric('Tramo de riesgo alto', str((portfolio.risk_band=='Alto').sum()))
    chosen_bands = st.multiselect('Tramos', ['Alto','Medio','Bajo'], default=['Alto','Medio','Bajo'])
    contract_options = sorted(portfolio.Contract.dropna().unique())
    contracts = st.multiselect('Contratos', contract_options, default=contract_options)
    filtered = portfolio[portfolio.risk_band.isin(chosen_bands) & portfolio.Contract.isin(contracts)]
    st.dataframe(filtered, use_container_width=True, hide_index=True)
    st.download_button('Exportar cartera filtrada (CSV)', data=filtered.to_csv(index=False).encode('utf-8-sig'),
                       file_name='cartera_priorizada.csv', mime='text/csv')

elif view == 'Detalle del cliente':
    st.subheader('Explicación individual de riesgo')
    customer = st.selectbox('Seleccionar cliente', portfolio.customerID.tolist())
    record = portfolio.loc[portfolio.customerID == customer].iloc[0]
    c1,c2,c3 = st.columns(3)
    c1.metric('Puntaje de riesgo', f"{record['score']:.3f}")
    c2.metric('Tramo', record['risk_band'])
    c3.metric('Instancia OOF', f"Fold {record['fold_id']}")
    local = explanations[explanations.customerID==customer].copy()
    local['impacto_absoluto'] = local.relative_contribution.abs()
    local = local.sort_values('impacto_absoluto', ascending=False)
    st.markdown('**Principales factores asociados al puntaje**')
    st.bar_chart(local.head(12).set_index('variable')['relative_contribution'], horizontal=True)
    st.dataframe(local[['variable','contribution','relative_contribution']], hide_index=True, use_container_width=True)
    st.caption('Los signos reflejan dirección en el espacio de salida del modelo; no implican causalidad. Contribución relativa = aporte consolidado dividido por suma de valores absolutos.')
    st.caption(f"Espacio de las contribuciones: {metadata['shap_space']}.")

elif view == 'Factores de la cartera':
    st.subheader('Relevancia agregada de factores')
    st.caption('Promedio del valor absoluto de la contribución relativa SHAP sobre los 7.043 clientes OOF.')
    st.bar_chart(factors.set_index('variable')['mean_abs_relative'], horizontal=True)
    st.dataframe(factors, hide_index=True, use_container_width=True)
    st.warning('Asociación predictiva ≠ causalidad. Las explicaciones OOF provienen de cinco instancias entrenadas independientemente.')

else:
    st.subheader('Evaluación final sobre el conjunto de prueba reservado')
    c1,c2,c3 = st.columns(3)
    c1.metric('Average Precision', f"{holdout['average_precision']:.3f}")
    c2.metric('ROC-AUC', f"{holdout['roc_auc']:.3f}")
    c3.metric('Brier', f"{holdout['brier']:.3f}")
    st.caption(f"Modelo trivial: AP={holdout['dummy_average_precision']:.3f}, ROC-AUC={holdout['dummy_roc_auc']:.3f}, Brier={holdout['dummy_brier']:.3f}.")
    st.markdown('**Precisión y exhaustividad (captura) acumuladas por decil**')
    plot = deciles.set_index('top_share_percent')[['precision_cumulative','recall_cumulative']]
    st.line_chart(plot)
    st.dataframe(deciles, use_container_width=True, hide_index=True)
    st.markdown('**Calibración observada**')
    st.scatter_chart(calibration, x='mean_predicted', y='fraction_positive')
    st.caption('Una curva cercana a la diagonal es compatible con mejor calibración; no implica validez temporal ni eficacia de intervenciones.')
    with st.expander('Diagnóstico descriptivo por grupos'):
        st.dataframe(pd.read_csv(ART/'holdout_group_diagnostics.csv'), hide_index=True, use_container_width=True)
        st.caption('AP depende de prevalencia; diferencias de grupo no prueban por sí solas discriminación ni equidad.')

st.divider()
st.caption('Prototipo académico local, sin CRM real, sin intervención comercial y sin inferencias causales.')
