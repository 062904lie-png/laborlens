/* Analytics is intentionally limited to data that this deployment records. */
let analyticsVersion=0;

function analyticsPeriodLabel(days){return Number(days)===7?'Last 7 days':'Last 30 days';}
function analyticsToday(){const parts=Object.fromEntries(new Intl.DateTimeFormat('en-US',{timeZone:'Asia/Manila',year:'numeric',month:'2-digit',day:'2-digit'}).formatToParts(new Date()).filter(part=>part.type!=='literal').map(part=>[part.type,part.value]));return new Date(Date.UTC(Number(parts.year),Number(parts.month)-1,Number(parts.day)));}
function analyticsIsoDate(value){return value.toISOString().slice(0,10);}
function analyticsDateFromISO(value){const parts=String(value||'').split('-').map(Number);return parts.length===3&&parts.every(Number.isFinite)?new Date(Date.UTC(parts[0],parts[1]-1,parts[2])):null;}
function analyticsFormatDate(value){const date=value instanceof Date?value:analyticsDateFromISO(value);return date?date.toLocaleDateString('en-US',{timeZone:'UTC',month:'short',day:'numeric',year:'numeric'}):'—';}
function analyticsPresetRange(days){const end=analyticsToday(),start=new Date(end);start.setUTCDate(end.getUTCDate()-Number(days)+1);return {from:analyticsIsoDate(start),to:analyticsIsoDate(end)};}
function analyticsRangeLabel(from,to){const start=analyticsDateFromISO(from),end=analyticsDateFromISO(to);if(!start||!end)return 'Selected dates';return `${analyticsFormatDate(start)} – ${analyticsFormatDate(end)}`;}
function analyticsRangeMatchesPreset(from,to,days){const preset=analyticsPresetRange(days);return from===preset.from&&to===preset.to;}
function analyticsPanel(title,body){return `<section class="panel analytics-prototype-panel"><div class="panel-hdr"><h2 class="panel-title">${esc(title)}</h2></div><div class="panel-body">${body}</div></section>`;}
function analyticsVerticalBars(rows){
  const safe=rows.slice(0,6),max=Math.max(1,...safe.map(r=>num(r.count))),colors=['#1765f5','#895cf0','#ffb01c','#a9bad0','#ed5468','#38b982'];
  if(!safe.length)return '<div class="empty">No stored records for this period.</div>';
  return `<div class="analytics-vbars" role="img" aria-label="Questions by language">${safe.map((r,i)=>`<div class="analytics-vbar"><strong>${num(r.count).toLocaleString()}</strong><i style="height:${Math.max(7,pct(r.count,max))}%;background:${colors[i%colors.length]}"></i><span>${esc(languageLabel(r.label))}</span></div>`).join('')}</div>`;
}
function analyticsCategoryTable(rows,total){
  const safe=rows;
  if(!safe.length)return '<div class="empty">No stored records for this period.</div>';
  return `<div class="analytics-category-table"><table aria-label="Questions by category"><colgroup><col class="category-rank"/><col class="category-name"/><col class="category-count"/><col class="category-share"/></colgroup><thead><tr><th>#</th><th>Category</th><th>Questions</th><th>%</th></tr></thead><tbody>${safe.map((row,index)=>`<tr><td>${index+1}</td><td title="${esc(row.label)}">${esc(row.label)}</td><td>${num(row.count).toLocaleString()}</td><td>${pct(row.count,total)}%</td></tr>`).join('')}</tbody></table></div>`;
}
function analyticsMetric(label,value,color,icon,sub){return `<article class="stat-card ${color}"><span class="stat-icon" data-enhanced="true">${AdminUI.icon(icon)}</span><div class="stat-lbl">${esc(label)}</div><div class="stat-val">${esc(value)}</div><small class="analytics-metric-sub">${esc(sub)}</small></article>`;}

function analyticsTrend(series){
  const dates=[...new Set(series.flatMap(s=>s.rows.map(r=>String(r.date).slice(0,10))))].sort();
  const values=series.map(s=>new Map(s.rows.map(r=>[String(r.date).slice(0,10),num(r.count)])));
  return AdminUI.line(series)+(dates.length?'<details class="analytics-data"><summary>View exact values by date</summary>'+AdminUI.table(['Date',...series.map(s=>esc(s.label))],dates.map(date=>[esc(date),...values.map(v=>(v.get(date)||0).toLocaleString())]))+'</details>':'');
}

function setupAnalytics(){
  const host=document.getElementById('tab-analytics'),defaultDays=localStorage.getItem('laborlens_default_report_period')==='7'?'7':'30',range=analyticsPresetRange(defaultDays),today=analyticsIsoDate(analyticsToday());
  host.innerHTML=`<div class="page-hdr analytics-page-header"><div class="page-hdr-left"><h1>${AdminUI.icon('chart')}Analytics</h1><p>Monitor system usage, user behavior, and performance metrics of LaborLens.</p></div><div class="analytics-header-actions"><div class="analytics-breadcrumb">Dashboard <span>›</span> <strong>Analytics</strong></div><div class="analytics-header-range" id="analytics-current-date">${AdminUI.icon('clock')}<span><strong>${analyticsRangeLabel(range.from,range.to)}</strong></span></div></div></div><form class="workbench-filters analytics-filters" id="analytics-filters"><label>Category<select name="category"><option value="">All Categories</option>${KB_CATEGORIES.map(c=>`<option>${esc(c)}</option>`).join('')}</select></label><label>Language<select name="language"><option value="">All Languages</option><option value="en">English</option><option value="fil">Filipino</option><option value="hil">Hiligaynon</option></select></label><label>Period<select id="reports-period"><option value="7" ${defaultDays==='7'?'selected':''}>Last 7 Days</option><option value="30" ${defaultDays==='30'?'selected':''}>Last 30 Days</option><option value="custom">Custom Date Range</option></select></label><fieldset class="analytics-date-filter"><legend>Date range</legend><div class="analytics-date-inputs"><label>From<input type="date" name="date_from" value="${range.from}" max="${today}" required/></label><label>To<input type="date" name="date_to" value="${range.to}" max="${today}" required/></label></div></fieldset><input type="hidden" name="days" value="${defaultDays}"/><button class="btn btn-primary" type="submit">${AdminUI.icon('search')} Filter</button><button class="btn" type="reset">↻ Reset</button></form><div id="analytics-results"></div>`;
  const form=host.querySelector('form');
  form.querySelector('.analytics-date-filter').outerHTML=`<label>From<input type="date" name="date_from" value="${range.from}" max="${today}" required/></label><label>To<input type="date" name="date_to" value="${range.to}" max="${today}" required/></label>`;
  form.querySelector('#reports-period').closest('label').firstChild.textContent='Quick range';
  form.querySelector('button[type="submit"]').classList.add('analytics-apply');
  form.querySelector('button[type="submit"]').innerHTML=`${AdminUI.icon('search')} Apply Filters`;
  const from=form.elements.date_from,to=form.elements.date_to;
  from.onchange=()=>{to.min=from.value||'';if(to.value&&from.value&&to.value<from.value)to.value=from.value;};
  from.onchange();
  document.getElementById('reports-period').onchange=e=>{if(e.target.value!=='custom')setAnalyticsPeriod(e.target.value);};
  form.onsubmit=e=>{e.preventDefault();loadAnalytics();};
  form.onreset=()=>setTimeout(()=>setAnalyticsPeriod(defaultDays),0);
}

function setAnalyticsPeriod(days){
  const form=document.getElementById('analytics-filters');
  if(!form)return;
  const range=analyticsPresetRange(days);
  form.elements.days.value=String(days);
  form.elements.date_from.value=range.from;
  form.elements.date_to.value=range.to;
  form.elements.date_to.min=range.from;
  loadAnalytics();
}

async function loadAnalytics(){
  if(!document.getElementById('analytics-results'))setupAnalytics();
  const v=++analyticsVersion,U=AdminUI,host=document.getElementById('analytics-results'),form=document.getElementById('analytics-filters'),formData=new FormData(form),from=String(formData.get('date_from')||''),to=String(formData.get('date_to')||''),days=String(formData.get('days')||7);
  if(!from||!to){host.innerHTML='<div class="err-msg">Choose both a From and To date.</div>';return;}
  if(from>to){host.innerHTML='<div class="err-msg">The From date must be on or before the To date.</div>';return;}
  const params=new URLSearchParams(formData),isPreset=analyticsRangeMatchesPreset(from,to,days),period=isPreset?days:'custom',rangeLabel=isPreset?analyticsPeriodLabel(days):analyticsRangeLabel(from,to);
  document.getElementById('reports-period').value=isPreset?days:'custom';
  document.getElementById('analytics-current-date').innerHTML=AdminUI.icon('clock')+`<span><strong>${esc(analyticsRangeLabel(from,to))}</strong></span>`;
  host.innerHTML='<div class="loading">Loading analytics...</div>';
  try{
    const d=await AdminPages.api('/admin/insights?'+params);
    if(v!==analyticsVersion)return;
    const totalFeedback=d.feedback.reduce((n,r)=>n+num(r.count),0),count=k=>num(d.feedback.find(r=>r.label===k)?.count),rate=k=>totalFeedback?pct(count(k),totalFeedback)+'%':'—',questionTotal=num(d.usage.questions),categoryTable=analyticsCategoryTable(d.categories,questionTotal);
    const feedbackSeries=['helpful','not_helpful','reported'].map(key=>({label:key.replaceAll('_',' '),rows:d.feedback_trend.filter(row=>row.label===key)}));
    const panels=[
      analyticsPanel('Questions Over Time',analyticsTrend([{label:'Questions',rows:d.volume}])),
      analyticsPanel('Questions by Language',analyticsVerticalBars(d.languages)),
      analyticsPanel('Questions by Category',`<div class="analytics-category-chart">${U.donut(d.categories)}${categoryTable}</div>`),
      analyticsPanel('Top Asked Topics','<p class="analytics-topic-note">Specific legal subjects identified from saved questions using the knowledge-base topic rules. Each question counts once.</p><div id="topic-ranking"></div><button class="btn ranking-actions" id="topics-all">View All Topics</button>'),
      analyticsPanel('User Activity',analyticsTrend([{label:'New users',rows:d.new_users},{label:'Active users',rows:d.active_users}])),
      analyticsPanel('Feedback Trend',analyticsTrend(feedbackSeries))
    ].join('');
    host.innerHTML=`<div class="analytics-kpis">${analyticsMetric('Total Questions',questionTotal.toLocaleString(),'blue','chat',rangeLabel)}${analyticsMetric('Active Users',num(d.usage.active_users).toLocaleString(),'green','users',rangeLabel)}${analyticsMetric('Helpful Rate',rate('helpful'),'green','check','Feedback records')}${analyticsMetric('Average Response Time',d.timing?.average==null||formData.get('language')?'Not recorded':(Number(d.timing.average)/1000).toFixed(1)+' s','purple','clock',formData.get('language')?'Response times are not recorded separately by language':'Based on saved response times')}</div><div class="analytics-prototype-grid">${panels}</div>`;
    let showAll=false;
    const ranked=Array.isArray(d.topics)?d.topics:[];
    function renderRanking(){document.getElementById('topic-ranking').innerHTML=U.table(['Rank','Topic','Questions','% of Questions','Action'],ranked.slice(0,showAll?ranked.length:5).map((r,i)=>[i+1,esc(r.label),r.count.toLocaleString(),pct(r.count,questionTotal)+'%',`<button class="btn btn-sm" data-topic-index="${i}">View Topic</button>`]));host.querySelectorAll('[data-topic-index]').forEach(b=>b.onclick=()=>openQuestionTopic(ranked[Number(b.dataset.topicIndex)]));}
    document.getElementById('topics-all').onclick=e=>{showAll=!showAll;e.currentTarget.textContent=showAll?'Show Top 5':'View All Topics';renderRanking();};
    renderRanking();
  }catch(e){if(v===analyticsVersion)host.innerHTML='<div class="err-msg">Unable to load analytics. <button class="btn" onclick="loadAnalytics()">Retry</button></div>';}
}
function openQuestionTopic(topic){
 const examples=Array.isArray(topic.examples)?topic.examples:[];
 AdminUI.drawer(topic.label,
   '<p>'+num(topic.count).toLocaleString()+' questions in the selected report filters.</p><p>Automatically grouped by question wording. A question covering several subjects is counted under one primary topic.</p>'+
   AdminUI.panel('Matching question examples',AdminUI.table(['Question','Times asked'],examples.map(row=>[esc(row.question),num(row.count).toLocaleString()])))+
   '<p>Showing up to five distinct examples.</p>');
}
function rankTopics(rows){
 const counts=new Map();
 for(const row of rows){const label=KB_CATEGORIES.includes(row.label)?row.label:'Others';counts.set(label,(counts.get(label)||0)+num(row.count));}
 return [...counts].map(([label,count])=>({label,count})).sort((a,b)=>b.count-a.count||a.label.localeCompare(b.label,'en'));
}
let topicDetailVersion=0;
async function openTopic(category,period){
 const v=++topicDetailVersion,params=new URLSearchParams(period);params.set('category',category);
 AdminUI.drawer(category==='Others'?'Other':category,'<div class="loading">Loading topic...</div>');
 try{
  const reviews=new URLSearchParams({category,language:params.get('language')||'',limit:5});
  for(const key of ['date_from','date_to'])if(params.get(key))reviews.set(key,params.get(key));
  const pending=new URLSearchParams(reviews);pending.set('needs_attention','true');pending.set('limit','1');
  const [data,questions,needs]=await Promise.all([AdminPages.api('/admin/insights?'+params),AdminPages.api('/admin/answer-reviews?'+reviews),AdminPages.api('/admin/answer-reviews?'+pending)]);
  if(v!==topicDetailVersion||!document.getElementById('doc-modal').classList.contains('show'))return;
  const U=AdminUI;
  document.getElementById('doc-modal-body').innerHTML=U.cards([['Questions',data.usage.questions],['Needs Review',needs.total]])+
    U.panel('Questions over time',U.line([{label:'Questions',rows:data.volume}]))+
    U.panel('Languages',U.bars(data.languages.map(r=>({...r,label:languageLabel(r.label)}))))+
    U.panel('Feedback (feedback events in selected period)',U.table(['Kind','Count'],data.feedback.map(r=>[esc(r.label),num(r.count)])))+
    U.panel('Recent answers in this topic',U.table(['Question','Action'],questions.items.map(r=>[esc(r.question),`<button class="btn btn-sm" data-topic-review="${Number(r.id)}">Review</button>`])));
  document.getElementById('doc-modal-foot').innerHTML='<button class="btn btn-primary" id="topic-questions">View Questions</button>';
  document.getElementById('topic-questions').onclick=async()=>{closeDocModal();await AdminShell.go('reviews');AdminReviews.filterTopic(category,Object.fromEntries(params));};
  document.querySelectorAll('[data-topic-review]').forEach(b=>b.onclick=async()=>{await AdminShell.go('reviews');AdminReviews.open(Number(b.dataset.topicReview));});
 }catch(e){if(v===topicDetailVersion)document.getElementById('doc-modal-body').innerHTML='<div class="err-msg">Unable to load this topic. Close and retry.</div>';}
}
