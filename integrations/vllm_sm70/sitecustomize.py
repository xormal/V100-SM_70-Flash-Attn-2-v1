# Copyright (c) 2026 xormal (Alexander Romanoff)
# SPDX-License-Identifier: MIT
#
# ТОЧКА, ГДЕ ВСТАЁТ ПОДМЕНА СКАНА GDN. Этот файл лежит в том же каталоге, что и пакет
# flash_attn_v100, и подъёмник кладёт этот каталог первым в PYTHONPATH. Значит:
#   * site.py импортирует sitecustomize из ЛЮБОГО каталога sys.path -> хук встаёт автоматически;
#   * PYTHONPATH наследуется spawn-воркерами vLLM -> хук встаёт и в них, а это принципиально:
#     скан считается ВНУТРИ воркера, и подмена только в процессе API-сервера не сделала бы ничего.
#
# ПОЧЕМУ НЕ ИЗ flash_attn_v100/__init__.py. Тот модуль импортируется ЛЕНИВО, когда vLLM выбирает
# бэкенд внимания -- то есть УЖЕ ПОСЛЕ того, как построена модель и qwen3_next связал у себя имя
# chunk_gated_delta_rule. Подмена оттуда опоздала бы ровно на один импорт и молча ничего не дала.
# Это и есть тот класс ошибки, ради которого гейт врезки обязан быть поведенческим.
#
# ОСТОРОЖНО И ТИХО. Ошибка здесь роняет ЛЮБОЙ python-процесс с этим PYTHONPATH, включая утилиты,
# не имеющие отношения к vLLM. Поэтому всё завёрнуто в широкий except и по умолчанию молчит.
import os

if os.environ.get("FA2SM70_GDN", "1") != "0":
    try:
        import fa2_gdn_shim

        fa2_gdn_shim.install()
    except Exception:  # noqa: BLE001
        if os.environ.get("FA2SM70_GDN_LOG", "0") == "1":
            import traceback

            traceback.print_exc()

# Тихий лог: клиентские ошибки валидации (длина промпта > контекста) -- одной строкой WARNING
# вместо 25-строчного ERROR-трейсбека. Ставится ЗДЕСЬ, потому что обёртка logging.config.dictConfig
# обязана появиться ДО импорта vllm.logger, который этот dictConfig и зовёт. Подробности и границы
# (что именно глушится, а что нет) -- в fa2sm70_logquiet.py.
if os.environ.get("FA2SM70_QUIET_CLIENT_ERR", "1") != "0":
    try:
        import fa2sm70_logquiet

        fa2sm70_logquiet.install()
    except Exception:  # noqa: BLE001
        pass

# Ловушка NaN (FA2SM70_NAN_TRAP=1): показывает ПЕРВЫЙ модуль, чей выход потерял конечность, и
# конечен ли при этом его вход. Нужна потому, что "забор из восклицаний" -- симптом на выходе, а
# источников у него за одну сессию нашлось три разных. По умолчанию выключена.
if os.environ.get("FA2SM70_NAN_TRAP", "0") == "1":
    try:
        import fa2sm70_nantrap

        fa2sm70_nantrap.install()
    except Exception:  # noqa: BLE001
        pass

# Сторож логитов (FA2SM70_LOGIT_WATCH=1): то же разделение "тело или голова", но БЕЗ хуков на
# модулях, то есть С ГРАФАМИ -- в том самом режиме, в котором сервер работает в бою. Ловушка выше
# требует --enforce-eager и потому меряет другой режим. По умолчанию выключен.
if os.environ.get("FA2SM70_LOGIT_WATCH", "0") == "1":
    try:
        import fa2sm70_logitwatch

        fa2sm70_logitwatch.install()
    except Exception:  # noqa: BLE001
        pass
