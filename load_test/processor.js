// Processor для Artillery нагрузочного тестирования
// Генерирует случайные task_id для тестирования

module.exports = {
  generateTaskId: function generateTaskId(context, events, done) {
    // Генерируем случайный UUID для task_id
    const uuid = 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
      const r = Math.random() * 16 | 0;
      const v = c === 'x' ? r : (r & 0x3 | 0x8);
      return v.toString(16);
    });
    
    context.vars.taskId = uuid;
    return done();
  }
};

