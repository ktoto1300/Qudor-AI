(function () {
  const replacements = [
    ['Режим поиска', 'Search mode'],
    ['Последовательный', 'Sequential'], ['Батчевый (быстрее)', 'Batched (faster)'],
    ['Батчевый собирает оценки раунда в один проход — быстрее, но это чуть другой поиск.', "Batched search evaluates a whole round at once — faster, but a slightly different search."],
    ['int8 квантование', 'int8 quantisation'],
    ['Грузит сеть в int8; на некоторых CPU медленнее — сравните время хода.', 'Loads the network as int8; slower on some CPUs — compare move time.'],
    ['Не удалось сменить режим поиска:', 'Could not change search mode:'],
    ['Не удалось переключить int8:', 'Could not toggle int8:'],
    ['Локальная игра и диагностика модели', 'Local game and model diagnostics'],
    ['Загрузка…', 'Loading…'], ['Доска Quoridor', 'Quoridor board'],
    ['Новая партия', 'New game'], ['Ход пешкой', 'Pawn move'],
    ['Стена ↔', 'Horizontal wall'], ['Стена ↕', 'Vertical wall'],
    ['Зелёным отмечены клетки, куда можно пойти.', 'Green cells show legal pawn moves.'],
    ['стены синего', 'Blue walls'], ['стены оранжевого', 'Orange walls'],
    ['Настройки игры', 'Game settings'], ['Режим', 'Mode'],
    ['Человек против ИИ', 'Human vs AI'], ['ИИ против ИИ', 'AI vs AI'],
    ['Загрузить модель', 'Load model'], ['Скорость воспроизведения', 'Playback speed'],
    ['Мощность ИИ на ход:', 'AI simulations per move:'], ['симуляции', 'simulations'],
    ['Больше симуляций — сильнее, но медленнее.', 'More simulations are stronger, but slower.'],
    ['Пауза', 'Pause'], ['Один шаг ИИ', 'One AI step'], ['Позиция', 'Position'],
    ['полуходов', 'plies'], ['Синий', 'Blue'], ['Оранжевый', 'Orange'],
    ['ходит', 'to move'], ['стен синего', 'Blue walls'], ['стен оранжевого', 'Orange walls'],
    ['оценка ИИ', 'AI evaluation'], ['время поиска', 'Search time'],
    ['Оценка ходов (поиск)', 'Move evaluation (search)'], ['История партии', 'Game history'],
    ['Метрики обучения', 'Training metrics'], ['Ошибка запроса', 'Request failed'],
    ['Победил синий', 'Blue wins'], ['Победил оранжевый', 'Orange wins'],
    ['Ничья: лимит', 'Draw: limit'], ['Клетка ', 'Cell '],
    ['Горизонтальная', 'Horizontal'], ['Вертикальная', 'Vertical'], ['стена ', 'wall '],
    ['Ход принят, ИИ думает…', 'Move accepted, AI is thinking…'], ['пешка →', 'pawn →'],
    ['стена ↔', 'horizontal wall'], ['стена ↕', 'vertical wall'], ['логит:', 'logit:'],
    ['Пока нет оценки', 'No evaluation yet'], ['ходов не показано', 'more moves hidden'],
    ['Партия не началась', 'Game has not started'], ['горизонтальные', 'horizontal'],
    ['вертикальные', 'vertical'], ['легальные', 'legal'], ['строк', 'rows'],
    ['Метрики не найдены', 'No metrics found'], ['Не удалось изменить мощность ИИ:', 'Could not change AI simulations:'],
    ['Модель загружена', 'Model loaded'], ['Модель не загружена:', 'Model not loaded:'],
    ['Снять паузу и дать ИИ продолжить', 'Resume and let AI continue'],
    ['Нет CSRF-токена — управление недоступно, перезагрузите страницу', 'No CSRF token — controls are unavailable; reload the page'],
    ['Checkpoint-файлы не найдены', 'No checkpoint files found'],
    ['Не удалось получить список моделей:', 'Could not load model list:'],
    ['Стартовая модель недоступна:', 'The default model is unavailable:'],
    ['Ваш ход', 'Your turn'], ['Ход ИИ', 'AI turn'], ['ИИ думает…', 'AI is thinking…'],
    ['Пока нет оценки', 'No evaluation yet'], ['Партия не началась', 'Game has not started']
  ];

  function translate(value) {
    let result = value;
    for (const [from, to] of replacements) result = result.split(from).join(to);
    return result;
  }

  function translateNode(node) {
    if (node.nodeType === Node.TEXT_NODE) {
      const next = translate(node.nodeValue);
      if (next !== node.nodeValue) node.nodeValue = next;
    } else if (node.nodeType === Node.ELEMENT_NODE) {
      for (const attr of ['aria-label', 'title']) {
        if (node.hasAttribute(attr)) node.setAttribute(attr, translate(node.getAttribute(attr)));
      }
      for (const child of node.childNodes) translateNode(child);
    }
  }

  function translatePage() { if (document.body) translateNode(document.body); }
  new MutationObserver(translatePage).observe(document.documentElement, { childList: true, subtree: true, characterData: true });
  document.addEventListener('DOMContentLoaded', translatePage);
})();
