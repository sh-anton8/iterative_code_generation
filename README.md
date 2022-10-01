# Итеративное написание кода с помощью Deep Learning

Процесс генерации текста существующими моделями DL отличается от поведения человека (модели генерируют сразу итоговый текст). В свежей статье https://arxiv.org/pdf/2208.11663.pdf предлагается архитектура, позволяющая генерировать текст итеративно (модель способна писать черновик,  добавлять предложения, предлагать правки и давать объяснения своим действиям). Данный проект адаптирует предложенный метод для генерации програмного кода. Такая модель могла бы переводить код из одного языка программирования на другой, а также вносить те или иные правки в существующий код по описанию коммита (без ручного вмешательства в код). 

