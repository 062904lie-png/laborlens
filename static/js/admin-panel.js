/* Presentation helpers only. Existing admin code owns requests and writes. */
(() => {
  const paths = {
    overview:'<rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/>',
    users:'<circle cx="9" cy="7" r="3"/><path d="M3 21v-3a6 6 0 0 1 12 0v3M16 4a3 3 0 0 1 0 6M17 13a5 5 0 0 1 4 5v3"/>',
    faqs:'<circle cx="12" cy="12" r="9"/><path d="M9 9a3 3 0 1 1 5 2c-2 1-2 2-2 3M12 17v.1"/>',
    knowledge:'<path d="M14 2H5v20h14V7Z M14 2v6h5M8 12h8M8 16h8"/>',
    analytics:'<rect x="3" y="13" width="3" height="8" rx="1"/><rect x="10" y="8" width="3" height="13" rx="1"/><rect x="17" y="3" width="3" height="18" rx="1"/>',
    performance:'<path d="M2 12h5l3-9 4 18 3-9h5"/>',
    queries:'<path d="M21 11a9 9 0 0 1-9 9H4l-2 2V11a9 9 0 0 1 19 0Z M7 11h.1M12 11h.1M17 11h.1"/>',
    check:'<circle cx="12" cy="12" r="9"/><path d="m7 12 3 3 7-7"/>',
    crown:'<path d="m3 6 4 5 5-8 5 8 4-5-2 13H5Z M5 22h14"/>',
    calendar:'<rect x="3" y="5" width="18" height="16" rx="2"/><path d="M7 2v6M17 2v6M3 11h18"/>',
    shield:'<path d="M12 2 3 6v7c0 5 9 9 9 9s9-4 9-9V6Z M12 6v12M7 9h10M7 9l-2 5h4ZM17 9l-2 5h4Z"/>'
  };
  const icon = key => `<svg viewBox="0 0 24 24" aria-hidden="true">${paths[key] || paths.analytics}</svg>`;
  const escape = value => String(value ?? '').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const count = value => Math.max(0, Number.isFinite(Number(value)) ? Number(value) : 0);
  const topicNames={wage_inquiry:'Pay and Wages',general_inquiry:'General Questions',needs_clarification:'Needs More Details','13th_month':'13th Month Pay'};
  const topic = value => String(topicNames[value] || value || 'General question').replace(/_/g,' ').replace(/\b\w/g,c=>c.toUpperCase());
  function dailySeries(rows) {
    return (Array.isArray(rows) ? rows : []).map(row=>({date:String(row.date || row.DATE || '').slice(0,10),count:count(row.count)})).sort((a,b)=>a.date.localeCompare(b.date));
  }
  function dailyChart(rows) {
    const data=dailySeries(rows);
    if(!data.length) return '<div class="empty">No daily question data yet.</div>';
    const max=Math.max(1,...data.map(r=>r.count));
    const ceiling=Math.max(4,Math.ceil(max/4)*4);
    const left=36,right=504,top=15,bottom=148;
    const points=data.map((row,i)=>({x:data.length===1?(left+right)/2:left+i*(right-left)/(data.length-1),y:bottom-row.count/ceiling*(bottom-top),...row}));
    const line=points.map(p=>`${p.x},${p.y}`).join(' ');
    const ticks=Array.from({length:5},(_,i)=>{const y=bottom-i*(bottom-top)/4;return `<line x1="${left}" x2="${right}" y1="${y}" y2="${y}" class="chart-grid"/><text x="26" y="${y+4}" text-anchor="end">${ceiling*i/4}</text>`;}).join('');
    const labels=points.filter((p,i)=>i===0 || i===points.length-1 || i%Math.max(1,Math.ceil(points.length/6))===0).map(p=>`<text x="${p.x}" y="173" text-anchor="middle">${escape(p.date.slice(5))}</text>`).join('');
    return `<div class="admin-chart"><svg viewBox="0 0 528 188" role="img" aria-label="Daily question counts. Exact dates and counts are listed in the table below."><defs><linearGradient id="admin-chart-fill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#2781ed" stop-opacity=".22"/><stop offset="100%" stop-color="#2781ed" stop-opacity=".025"/></linearGradient></defs>${ticks}${data.length>1?`<polygon points="${points[0].x},${bottom} ${line} ${points.at(-1).x},${bottom}" fill="url(#admin-chart-fill)"/><polyline points="${line}" fill="none" stroke="#1875ed" stroke-width="2.5"/>`:''}${points.map(p=>`<circle cx="${p.x}" cy="${p.y}" r="3.5" fill="#1875ed"><title>${escape(p.date)}: ${p.count} questions</title></circle>`).join('')}${labels}</svg></div><div class="tbl-wrap"><table><thead><tr><th>Date</th><th>Questions</th></tr></thead><tbody>${data.map(row=>`<tr><td>${escape(row.date)}</td><td class="td-strong">${row.count.toLocaleString()}</td></tr>`).join('')}</tbody></table></div>`;
  }
  function renderOverview(data) {
    const rows=Array.isArray(data.intent_distribution)?data.intent_distribution:[];
    const max=Math.max(1,...rows.map(r=>count(r.count)));
    document.getElementById('intent-chart').innerHTML=rows.length?`<div class="tbl-wrap"><table class="admin-topic-table"><thead><tr><th>Topic</th><th aria-label="Relative number of questions"></th><th>Questions</th></tr></thead><tbody>${rows.map(r=>`<tr><td>${escape(topic(r.intent))}</td><td><div class="admin-topic-track"><span style="width:${count(r.count)/max*100}%"></span></div></td><td><span class="badge badge-blue">${count(r.count).toLocaleString()}</span></td></tr>`).join('')}</tbody></table></div>`:'<div class="empty">No questions yet.</div>';
    document.getElementById('vol-table').innerHTML=dailyChart(data.daily_volume);
  }
  let lastUsers=null,currentPage=1;
  const pageSize=4;
  function userPage(users) {
    if(users!==lastUsers){lastUsers=users;currentPage=1;}
    const pages=Math.max(1,Math.ceil(users.length/pageSize));
    currentPage=Math.min(currentPage,pages);
    const offset=(currentPage-1)*pageSize;
    let footer=document.getElementById('admin-user-pagination');
    if(!footer){footer=document.createElement('div');footer.id='admin-user-pagination';document.getElementById('users-tbody').closest('.panel').append(footer);}
    footer.innerHTML=`<span>${users.length?`Showing ${offset+1}–${Math.min(offset+pageSize,users.length)} of ${users.length} users`:'No matching users'}</span><div><button type="button" data-page="${currentPage-1}" aria-label="Previous user page" ${currentPage===1?'disabled':''}>‹</button><span class="admin-page-number">${currentPage} / ${pages}</span><button type="button" data-page="${currentPage+1}" aria-label="Next user page" ${currentPage===pages?'disabled':''}>›</button></div>`;
    footer.querySelectorAll('button').forEach(button=>button.addEventListener('click',()=>{currentPage=Number(button.dataset.page);window.renderUsers(lastUsers);}));
    return {rows:users.slice(offset,offset+pageSize),offset};
  }
  function syncNavigation(name) {
    document.querySelectorAll('#left-nav [data-tab]').forEach(button=>button.setAttribute('aria-current',button.dataset.tab===name?'page':'false'));
  }
  function enhanceCards() {
    const keys={Users:'users',Chats:'queries',Documents:'knowledge',Review:'check',Queries:'queries',Sessions:'queries',Feedback:'check',Speed:'performance',Slow:'performance',Total:'analytics'};
    document.querySelectorAll('#dashboard .stat-icon:not([data-enhanced])').forEach(el=>{const key=keys[el.textContent.trim()]||'analytics';el.innerHTML=icon(key);el.dataset.enhanced='true';});
  }
  function setup() {
    document.querySelectorAll('#left-nav .nav-item').forEach(el=>{
      el.querySelector('.nav-icon').innerHTML=icon(el.dataset.tab);
      el.setAttribute('role','button');el.tabIndex=0;
      el.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();el.click();}});
    });
    document.querySelectorAll('#dashboard .form-group').forEach(group=>{const label=group.querySelector('label');const field=group.querySelector('input[id],select[id],textarea[id]');if(label && field && !label.htmlFor)label.htmlFor=field.id;});
    syncNavigation('overview');
    const seal=document.querySelector('#topbar .tb-seal');if(seal)seal.innerHTML=icon('shield');
    const date=new Intl.DateTimeFormat('en',{weekday:'long',month:'short',day:'numeric',year:'numeric'}).format(new Date());
    document.querySelectorAll('#dashboard .page-hdr').forEach(header=>{const el=document.createElement('div');el.className='admin-date';el.innerHTML=`${icon('calendar')}<div><time>${escape(date)}</time><span>Welcome back.</span></div>`;header.append(el);});
    const metrics=[['users','All registered accounts'],['check','Accounts with access'],['crown','Accounts with admin access']];
    document.querySelectorAll('#tab-users .mini-metric').forEach((el,i)=>{el.classList.add('admin-user-metric',`user-metric-${i}`);el.insertAdjacentHTML('afterbegin',`<span class="admin-metric-icon">${icon(metrics[i][0])}</span>`);el.insertAdjacentHTML('beforeend',`<p>${metrics[i][1]}</p>`);});
    const panelIcons=['analytics','analytics','users','faqs','faqs','knowledge'];
    document.querySelectorAll('#dashboard .panel-title').forEach((el,i)=>{el.textContent=el.textContent.replace(/^[^A-Za-z0-9]+/,'');el.insertAdjacentHTML('afterbegin',`<span class="admin-panel-icon">${icon(panelIcons[i]||'knowledge')}</span>`);});
    enhanceCards();
    const observer=new MutationObserver(enhanceCards);
    document.querySelectorAll('#stat-grid,#analytics-stats,#perf-stats').forEach(el=>observer.observe(el,{childList:true,subtree:true}));
  }
  window.LaborLensAdminUI={renderOverview,userPage,syncNavigation,dailySeries,dailyChart};
  setup();
})();
