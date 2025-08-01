# Задача 2. TF-IDF Baseline

### Получились следующие метрики:

Recall@1: 0.363645 — в 36.36% случаев правильный документ находится на 1 месте выдачи (показатель хорошей способности выявления ключевых терминов, характерных для конкретных пар вопрос-ответ)

Recall@3: 0.546915 — чуть более, чем в половине случаев правильный ответ содержится в топ-3 результатах (хорошая ранжировка документов по релевантности)

Recall@10: 0.711628 — в 71% случаев документ содержится в топ-10 (эффективная фильтрация явно нерелевантных документов при большом размере выдачи)

MRR: 0.473967 — в среднем правильный ответ находится между 2 и 3 позициями (1/0.4739 = 2.11), достаточно хороший результат

### Ограничения текущего подхода:

- Возможные проблемы с памятью и производительностью:  
  sim_matrix = cosine_similarity(X_questions, X_docs) требует O(N*M) памяти (N - вопросы теста, M - документы теста -> ~80K * 80K)

- Проблемы TF-IDF:  
  возможны проблемы с учётом семантики (похожие запросы могут не иметь пересечений в терминах)  
  синонимы обрабатываются как разные термины  
  слабая адаптация к длинным текстам: длинные ответы получают зашумленные вектора

# Задача 3. E5 Baseline

### Получились следующие метрики:

Recall@1: 0.6936 — в 69.36% верный документ находится на 1 месте выдачи

Recall@3: 0.8914 — в почти 90% случаях правильный ответ находится в топ-3 выдачи — высокая плотность релевантных документов в верхней части выдачи

Recall@10: 0.9689 — почти все правильные ответы содержатся в топ-10 результатов

MRR: 0.7984 — в большинстве случаев правильный документ находится на 1-2 позиции (1/0.7984 = 1.25)

### Сравнение с TF-IDF

Метрика TF-IDF E5-Base Улучшение  
Recall@1 36.36% 69.36% +33%  
Recall@3 54.69% 89.14% +34%  
Recall@10 71.16% 96.89% +25%  
MRR 47.40% 79.84% +32%

В сравнении с TF-IDF алгоритм показал улучшение работы: по всем метрикам наблюдается прирост на 25-34%. Это связано с архитектурными преимуществами, разделением ролей текстов (query и passage), а также тем, что E5-Base учитывает не только лексические совпадения, но и контекст и смысл текста

# Задача 4. E5 Train

### Получились следующие метрики:

Contrastive Model Metrics:  
Recall@1: 0.5599 — В 56% случаев правильный ответ на первом месте  
Recall@3: 0.7627 — В 76% случаев ответ в топ-3  
Recall@10: 0.8847 — 88% случаев правильный ответ в топ-10  
MRR: 0.6738 — Средний ранг правильного ответа ~1.48

Triplet Model Metrics:  
Recall@1: 0.1362 — В 13% случаев правильный ответ на первом месте  
Recall@3: 0.2270 — В 23% случаев ответ в топ-3  
Recall@10: 0.3464 — в 35% случаев ответ в топ-10  
MRR: 0.1963 — В среднем правильный ответ на 5 позиции

Contrastive Loss показал себя значительно лучше по всем метрикам: на Recall@1 разница составляет +43%, MRR выше на +48%

### Сравнение constractive и triplet loss

Contrastive Loss явно разделяет позитивные (близкие) и негативные (далекие) пары  
Contrastive Loss менее чувствителен к "плохим" негативам, так как учится явно разделять классы  
Triplet Loss теряет эффективность, если негативы слишком далеки от анкора

### Сравнение с ванильным E5

Текущие результаты получились хуже, чем с ванильным E5 (например, Recall@1 0.6936 против 0.56 и 0.13; Recall@3 0.89 против 0.76 и 0.22)

Вероятные причины: случайные негативы ухудшают качество эмбеддингов (они слишком просты, что снижает качество), возможно перебучение

# Задача 5. E5 Train with Hard Negatives

### Получились следующие метрики:

Recall@1: 0.6889 — почти в 70% случаев нужный документ находится на 1-м месте  
Recall@3: 0.8883 — в 89% случаев ответ в топ-3  
Recall@10: 0.9663 — почти всегда правильный ответ попадает в топ-10  
MRR: 0.7935 — в среднем правильный ответ находится на 1-2 позициях

### Сравнение с random negatives:

Использование hard negatives показало заметное улучшение точности модели в сравнении с random negatives, итоговые результаты по метрикам сравнимы с ванильным E5

Метрика Hard Negatives Random Negatives Улучшение  
Recall@1 68.89% 38.61% +30.28%  
Recall@3 88.83% 55.29% +33.54%  
Recall@10 96.63% 66.92% +29.71%  
MRR 79.35% 48.14% +31.21%

Прирост связан с тем, что hard negatives документы семантически ближе к запросу, что позволяет модели лучше понимать различия между похожими документами, повышается качество разделения классов, снижен риск переобучения
