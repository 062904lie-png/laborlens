/* Independent FAQ reader: does not change the consultation or message state. */
(() => {
  const trigger = document.getElementById('open-faq-browser');
  if (!trigger) return;
  const dialog = document.createElement('dialog');
  dialog.id = 'faq-browser';
  dialog.setAttribute('aria-labelledby', 'faq-browser-title');
  dialog.innerHTML = `<header><h2 id="faq-browser-title">Frequently Asked Questions</h2><button type="button" data-close aria-label="Close frequently asked questions">Close</button></header>
    <p>Browse answers reviewed and published by LaborLens administrators. Check the supporting sources and whether the answer applies to your circumstances.</p>
    <div class="faq-filters"><label>Search questions<input type="search" data-search maxlength="200" placeholder="Search questions or answers…"></label><label>Category<select data-category><option value="">All categories</option></select></label><label>Language<select data-language><option value="">All languages</option><option value="en">English</option><option value="fil">Filipino</option><option value="hil">Hiligaynon</option></select></label></div>
    <p data-status role="status" aria-live="polite"></p><div data-results></div>`;
  document.body.append(dialog);
  const $ = selector => dialog.querySelector(selector);
  const languageNames = {en:'English',fil:'Filipino',hil:'Hiligaynon'};
  let records = [], controller, loaded = false;
  function textElement(tag, text, className) {
    const element = document.createElement(tag);
    element.textContent = text || '';
    if (className) element.className = className;
    return element;
  }
  function render() {
    if (!loaded) return;
    const query = $('[data-search]').value.trim().toLowerCase();
    const category = $('[data-category]').value, language = $('[data-language]').value;
    const rows = records.filter(row => (!category || (row.category || 'Uncategorized') === category) && (!language || row.language === language) && [row.question,row.answer,row.category,row.source_citation].join(' ').toLowerCase().includes(query));
    $('[data-status]').textContent = rows.length ? `${rows.length} published ${rows.length === 1 ? 'answer' : 'answers'}` : records.length ? 'No FAQs match these filters.' : 'No FAQs have been published yet. You can continue using the chat.';
    $('[data-results]').replaceChildren();
    for (const row of rows) {
      const details = document.createElement('details');
      details.append(textElement('summary', row.question), textElement('p', `${row.category || 'Uncategorized'} · ${languageNames[row.language] || row.language}`, 'faq-meta'), textElement('p',row.answer,'faq-answer'));
      details.append(textElement('h3','Supporting sources'));
      const sources = document.createElement('ul');
      for (const source of row.sources || []) {
        const item = textElement('li',[source.title,source.law_number,source.article_section].filter(Boolean).join(' — '));
        try {
          const url = new URL(source.source_url);
          if (['https:','http:'].includes(url.protocol) && !url.username && !url.password) {
            const link = textElement('a',' Open source');
            link.href = url.href; link.target = '_blank'; link.rel = 'noopener noreferrer';
            item.append(link);
          }
        } catch (_) { /* A source may have no public URL. Its citation remains visible. */ }
        sources.append(item);
      }
      details.append(sources);
      if (row.source_citation) details.append(textElement('p',row.source_citation,'faq-citation'));
      $('[data-results]').append(details);
    }
  }
  async function load() {
    controller?.abort();
    const request = new AbortController(); controller = request;
    loaded = false; records = []; $('[data-results]').replaceChildren();
    $('[data-category]').replaceChildren(new Option('All categories',''));
    $('[data-status]').textContent = 'Loading published FAQs…';
    try {
      const response = await fetch(`${window.LABORLENS_API_URL}/faqs`,{cache:'no-store',signal:request.signal});
      if (!response.ok) throw new Error('FAQ request failed');
      const data = await response.json();
      if (!Array.isArray(data)) throw new Error('Invalid FAQ response');
      if (request.signal.aborted) return;
      records = data; loaded = true;
      for (const category of [...new Set(records.map(row=>row.category||'Uncategorized'))].sort()) $('[data-category]').append(new Option(category,category));
      render();
    } catch (error) {
      if (request.signal.aborted) return;
      $('[data-status]').textContent = 'FAQs could not be loaded. Please try again.';
      const retry = textElement('button','Retry'); retry.type='button'; retry.onclick=load;
      $('[data-results]').replaceChildren(retry);
    }
  }
  trigger.addEventListener('click',()=>{ if (!dialog.open) dialog.showModal(); load(); });
  $('[data-close]').onclick=()=>dialog.close();
  dialog.addEventListener('close',()=>{controller?.abort();trigger.focus();});
  $('[data-search]').oninput=render;
  $('[data-category]').onchange=render;
  $('[data-language]').onchange=render;
})();
