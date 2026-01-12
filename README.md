# ML7

Архитектура shadow deployment: <br>

```
class ShadowDeploymentSystem:
    
    def __init__(self, prod_model, challenger_model):
        """
        Инициализация с prod и challenger моделями
        """

        self.prod_model = prod_model
        self.challenger_model = challenger_model
        self.request_logs = []
        
        print(f" Shadow Deployment система инициализирована")
        print(f"  PROD: {type(prod_model).__name__}")
        print(f"  CHALLENGER: {type(challenger_model).__name__}")
    
    def handle_request(self, features, request_id=None):
        """
        Обработка входящего запроса
        """

        if request_id is None:
            request_id = f"req_{len(self.request_logs):04d}"
        
        # 1. PROD предсказание
        prod_pred = int(self.prod_model.predict([features])[0])
        
        # 2. CHALLENGER предсказание 
        challenger_pred = int(self.challenger_model.predict([features])[0])
        
        # 3. Уверенности предсказаний
        prod_proba = self.prod_model.predict_proba([features])[0]
        challenger_proba = self.challenger_model.predict_proba([features])[0]
        
        # 4. Логи
        log_entry = {
            'request_id': request_id,
            'timestamp': datetime.now().strftime("%H:%M:%S.%f")[:-3],
            'prod_prediction': prod_pred,
            'prod_confidence': float(max(prod_proba)),
            'challenger_prediction': challenger_pred,
            'challenger_confidence': float(max(challenger_proba)),
            'has_discrepancy': prod_pred != challenger_pred,
            'features': features.tolist()  # Сохраняем для анализа
        }
        
        # 5. Сохраняем лог
        self.request_logs.append(log_entry)
        
        # 6. Возвращаем только PROD предсказание
        return prod_pred
    
    def simulate_traffic(self, features_list):
        """
        Симуляция потока запросов
        """

        print(f" Симуляция {len(features_list)} запросов...")
        
        for i, features in enumerate(features_list):
            request_id = f"sim_{i:04d}"
            self.handle_request(features, request_id)
            
            # Выводим прогресс для первых 5 запросов
            if i < 5:
                latest_log = self.request_logs[-1]
                if latest_log['has_discrepancy']:
                    print(f"  {request_id}:  РАСХОЖДЕНИЕ! PROD: {latest_log['prod_prediction']}, CHALLENGER: {latest_log['challenger_prediction']}")
                else:
                    print(f"  {request_id}: ✓ СОВПАДЕНИЕ")
        
        print(f" Обработано {len(features_list)} запросов")
        print(f" В логах: {len(self.request_logs)} записей")
    
    def get_metrics(self):
        """
        Вычисление метрик на основе логов
        """

        if not self.request_logs:
            return {"error": "Нет данных в логах"}
        
        df = pd.DataFrame(self.request_logs)
        
        # Основные метрики
        total_requests = len(df)
        discrepancies = df['has_discrepancy'].sum()
        discrepancy_rate = discrepancies / total_requests
        
        # Уверенность моделей
        avg_prod_conf = df['prod_confidence'].mean()
        avg_challenger_conf = df['challenger_confidence'].mean()
        
        # Анализ расхождений
        discrepancy_analysis = {}
        if discrepancies > 0:
            discrepancy_df = df[df['has_discrepancy'] == True]
            
            # Кто был увереннее при расхождениях
            challenger_more_confident = (discrepancy_df['challenger_confidence'] > 
                                        discrepancy_df['prod_confidence']).sum()
            prod_more_confident = discrepancies - challenger_more_confident
            
            discrepancy_analysis = {
                'challenger_more_confident': challenger_more_confident,
                'prod_more_confident': prod_more_confident,
                'challenger_confidence_win_rate': challenger_more_confident / discrepancies
            }
        
        metrics = {
            'total_requests': total_requests,
            'discrepancies': int(discrepancies),
            'discrepancy_rate': discrepancy_rate,
            'avg_confidence': {
                'prod': avg_prod_conf,
                'challenger': avg_challenger_conf
            },
            'discrepancy_analysis': discrepancy_analysis
        }
        
        return metrics
    
    def compare_predictions(self):
        """
        Сравнение предсказаний двух моделей
        """

        if not self.request_logs:
            print(" Нет данных для сравнения")
            return None
        
        df = pd.DataFrame(self.request_logs)
        
        print("="*50)
        print(" СРАВНЕНИЕ ПРЕДСКАЗАНИЙ PROD vs CHALLENGER")
        print("="*50)
        
        # Подсчёт совпадений/расхождений
        match_count = (~df['has_discrepancy']).sum()
        mismatch_count = df['has_discrepancy'].sum()
        
        print(f"\n Совпадения: {match_count} ({match_count/len(df):.1%})")
        print(f" Расхождения: {mismatch_count} ({mismatch_count/len(df):.1%})")
        
        # Распределение предсказаний
        print(f"\n Распределение предсказаний:")
        print(f"  PROD 1s: {df['prod_prediction'].sum()} ({df['prod_prediction'].mean():.1%})")
        print(f"  CHALLENGER 1s: {df['challenger_prediction'].sum()} ({df['challenger_prediction'].mean():.1%})")
        
        # Уверенность
        print(f"\n Уверенность моделей:")
        print(f"  PROD средняя: {df['prod_confidence'].mean():.3f}")
        print(f"  CHALLENGER средняя: {df['challenger_confidence'].mean():.3f}")
        
        return df
```


Сохранение метрик и логов: <br>

```
print("\n ЭКСПОРТ РЕЗУЛЬТАТОВ...")

# Сохраняем логи
logs_df = pd.DataFrame(shadow_system.request_logs)
logs_df.to_csv('shadow_deployment_logs.csv', index=False)

# Сохраняем вывод по метрикам
with open('shadow_deployment_metrics.txt', 'w') as f:
    f.write("="*50 + "\n")
    f.write("SHADOW DEPLOYMENT METRICS REPORT\n")
    f.write("="*50 + "\n\n")
    
    f.write(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write("МОДЕЛИ:\n")
    f.write(f"  PROD: {type(prod_model).__name__}\n")
    f.write(f"  CHALLENGER: {type(challenger_model).__name__}\n\n")
    
    f.write("ТОЧНОСТЬ НА ТЕСТЕ:\n")
    f.write(f"  PROD: {prod_test_accuracy:.3f}\n")
    f.write(f"  CHALLENGER: {challenger_test_accuracy:.3f}\n")
    f.write(f"  Разница: {challenger_test_accuracy - prod_test_accuracy:+.3f}\n\n")
    
    f.write("SHADOW DEPLOYMENT МЕТРИКИ:\n")
    f.write(f"  Всего запросов: {metrics['total_requests']}\n")
    f.write(f"  Расхождений: {metrics['discrepancies']}\n")
    f.write(f"  Уровень расхождений: {metrics['discrepancy_rate']:.2%}\n\n")
    
    f.write("РЕКОМЕНДАЦИЯ:\n")
    if all_criteria_met:
        f.write("  DEPLOYMENT РЕКОМЕНДУЕТСЯ\n")
        f.write("  CHALLENGER модель готова заменить PROD\n")
    else:
        f.write("  DEPLOYMENT НЕ РЕКОМЕНДУЕТСЯ\n")
        f.write("  Требуется доработка CHALLENGER модели\n")

```

<img width="469" height="174" alt="image" src="https://github.com/user-attachments/assets/a4a8a4da-570d-4cb6-9b26-4b57d4d852b0" /> <br>

<img width="1515" height="552" alt="image" src="https://github.com/user-attachments/assets/54a4216f-553e-448e-ad46-1acb305b8777" /> <br>

<img width="725" height="687" alt="image" src="https://github.com/user-attachments/assets/372bce1d-a76a-4d26-adbd-5b80d19e09b7" /> <br>

<br>

Сравнение и анализ данных: <br>
```
print("\n" + "="*50)
print(" АНАЛИЗ РЕЗУЛЬТАТОВ SHADOW DEPLOYMENT")
print("="*50)

# Получаем метрики
metrics = shadow_system.get_metrics()

print(f"\n ОСНОВНЫЕ МЕТРИКИ:")
print(f"  Всего запросов: {metrics['total_requests']}")
print(f"  Расхождений: {metrics['discrepancies']}")
print(f"  Уровень расхождений: {metrics['discrepancy_rate']:.2%}")

print(f"\n УВЕРЕННОСТЬ МОДЕЛЕЙ:")
print(f"  PROD средняя уверенность: {metrics['avg_confidence']['prod']:.3f}")
print(f"  CHALLENGER средняя уверенность: {metrics['avg_confidence']['challenger']:.3f}")

# Детальный анализ расхождений
if metrics['discrepancy_analysis']:
    analysis = metrics['discrepancy_analysis']
    print(f"\n АНАЛИЗ РАСХОЖДЕНИЙ:")
    print(f"  CHALLENGER был увереннее в {analysis['challenger_more_confident']} случаях")
    print(f"  PROD был увереннее в {analysis['prod_more_confident']} случаях")
    print(f"  CHALLENGER win rate по уверенности: {analysis['challenger_confidence_win_rate']:.1%}")

# Сравнение предсказаний
df_comparison = shadow_system.compare_predictions()
```

ИТОГ: <br>
```
==================================================
 АНАЛИЗ РЕЗУЛЬТАТОВ SHADOW DEPLOYMENT
==================================================

 ОСНОВНЫЕ МЕТРИКИ:
  Всего запросов: 100
  Расхождений: 8
  Уровень расхождений: 8.00%

 УВЕРЕННОСТЬ МОДЕЛЕЙ:
  PROD средняя уверенность: 0.849
  CHALLENGER средняя уверенность: 0.846

 АНАЛИЗ РАСХОЖДЕНИЙ:
  CHALLENGER был увереннее в 3 случаях
  PROD был увереннее в 5 случаях
  CHALLENGER win rate по уверенности: 37.5%
==================================================
 СРАВНЕНИЕ ПРЕДСКАЗАНИЙ PROD vs CHALLENGER
==================================================

 Совпадения: 92 (92.0%)
 Расхождения: 8 (8.0%)

 Распределение предсказаний:
  PROD 1s: 50 (50.0%)
  CHALLENGER 1s: 48 (48.0%)

 Уверенность моделей:
  PROD средняя: 0.849
  CHALLENGER средняя: 0.846
```

Анализ для деплоя: <br>
```
print("\n" + "="*50)
print(" ПРИНЯТИЕ РЕШЕНИЯ О DEPLOYMENT CHALLENGER МОДЕЛИ")
print("="*50)

# Оценка качества на тестовых данных (с ground truth)
print("\n ОЦЕНКА КАЧЕСТВА НА ТЕСТОВЫХ ДАННЫХ:")

# Вычисляем accuracy на тестовых данных
prod_test_predictions = prod_model.predict(test_requests)
challenger_test_predictions = challenger_model.predict(test_requests)

prod_test_accuracy = (prod_test_predictions == test_labels).mean()
challenger_test_accuracy = (challenger_test_predictions == test_labels).mean()

print(f"  PROD accuracy: {prod_test_accuracy:.3f}")
print(f"  CHALLENGER accuracy: {challenger_test_accuracy:.3f}")
print(f"  Разница: {challenger_test_accuracy - prod_test_accuracy:+.3f}")

# Shadow deployment метрики
print(f"\n SHADOW DEPLOYMENT МЕТРИКИ:")
print(f"  Уровень расхождений: {metrics['discrepancy_rate']:.2%}")
print(f"  PROD уверенность: {metrics['avg_confidence']['prod']:.3f}")
print(f"  CHALLENGER уверенность: {metrics['avg_confidence']['challenger']:.3f}")

if metrics['discrepancy_analysis']:
    analysis = metrics['discrepancy_analysis']
    print(f"  CHALLENGER увереннее при расхождениях: {analysis['challenger_confidence_win_rate']:.1%}")

# Критерии для принятия решения
print("\n КРИТЕРИИ ДЛЯ DEPLOYMENT:")
criteria = []

# Критерий 1: CHALLENGER должен быть точнее
is_challenger_more_accurate = challenger_test_accuracy > prod_test_accuracy
criteria.append(("CHALLENGER точнее PROD", is_challenger_more_accurate, 
                f"{challenger_test_accuracy:.3f} > {prod_test_accuracy:.3f}"))

# Критерий 2: Уровень расхождений < 30%
acceptable_discrepancy_rate = metrics['discrepancy_rate'] < 0.3
criteria.append(("Уровень расхождений < 30%", acceptable_discrepancy_rate,
                f"{metrics['discrepancy_rate']:.1%}"))

# Критерий 3: CHALLENGER увереннее при расхождениях
if metrics['discrepancy_analysis']:
    is_challenger_more_confident = analysis['challenger_confidence_win_rate'] > 0.5
    criteria.append(("CHALLENGER увереннее при расхождениях", is_challenger_more_confident,
                    f"{analysis['challenger_confidence_win_rate']:.1%} > 50%"))
else:
    criteria.append(("Нет расхождений для анализа", True, "Нет данных"))

print("\n ПРОВЕРКА КРИТЕРИЕВ:")
all_criteria_met = True
for name, met, details in criteria:
    status = "YES" if met else "NO"
    print(f"  {status} {name}: {details}")
    if not met:
        all_criteria_met = False

print("\n" + "="*50)
print(" ФИНАЛЬНОЕ РЕШЕНИЕ:")
print("="*50)

if all_criteria_met:
    print("\n DEPLOYMENT РЕКОМЕНДУЕТСЯ!")
else:
    print("\n DEPLOYMENT НЕ РЕКОМЕНДУЕТСЯ!")
    
```

ИТОГ: <br>
```
==================================================
 ПРИНЯТИЕ РЕШЕНИЯ О DEPLOYMENT CHALLENGER МОДЕЛИ
==================================================

 ОЦЕНКА КАЧЕСТВА НА ТЕСТОВЫХ ДАННЫХ:
  PROD accuracy: 0.830
  CHALLENGER accuracy: 0.870
  Разница: +0.040

 SHADOW DEPLOYMENT МЕТРИКИ:
  Уровень расхождений: 8.00%
  PROD уверенность: 0.849
  CHALLENGER уверенность: 0.846
  CHALLENGER увереннее при расхождениях: 37.5%

 КРИТЕРИИ ДЛЯ DEPLOYMENT:

ПРОВЕРКА КРИТЕРИЕВ:
  YES CHALLENGER точнее PROD: 0.870 > 0.830
  YES Уровень расхождений < 30%: 8.0%
  NO CHALLENGER увереннее при расхождениях: 37.5% > 50%

==================================================
 ФИНАЛЬНОЕ РЕШЕНИЕ:
==================================================

 DEPLOYMENT НЕ РЕКОМЕНДУЕТСЯ!

```











