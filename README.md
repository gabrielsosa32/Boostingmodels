# 🔍 Comparación de Modelos Boosting en Machine Learning

Este proyecto presenta una comparación entre tres algoritmos de ensamble secuencial: **AdaBoost**, **Gradient Boosting** y **XGBoost**, aplicados a un problema de clasificación binaria. Se entrena cada modelo con hiperparámetros por defecto y luego se realiza una búsqueda de hiperparámetros para mejorar su rendimiento.

---

## 📁 Contenido

- Preprocesamiento de datos
- Entrenamiento de modelos (default y tuned)
- Evaluación con métricas:
  - Accuracy
  - F1-Score
  - AUC (Área bajo la curva ROC)
- Curvas ROC para cada modelo
- Comparación visual de resultados
- Conclusiones

---

## 🔧 Tecnologías utilizadas

- Python 3.x
- Pandas
- Scikit-learn
- XGBoost
- Matplotlib / Seaborn

---

## 📊 Resultados Finales

| Modelo             | Accuracy | F1-Score | AUC   |
|--------------------|----------|----------|--------|
| AdaBoost (tuned)         | 0.8059   | 0.8083   | 0.8907 |
| Gradient Boosting (tuned)| 0.8017   | 0.8000   | 0.8928 |
| XGBoost (tuned)          | 0.8017   | 0.7983   | 0.8879 |

> **Conclusión**: Todos los modelos mejoraron su desempeño tras el ajuste de hiperparámetros, siendo Gradient Boosting el que obtuvo el mejor AUC general.

---

## 🚀 Cómo usar este proyecto

1. Clona este repositorio:
   ```bash
   git clone https://github.com/tuusuario/nombre-del-repo.git
