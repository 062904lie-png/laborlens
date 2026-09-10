/* Reports → Feedback. Existing feedback-page and insights API results only. */
const feedbackKinds=[
 {key:'helpful',label:'Helpful',tone:'green',icon:'check'},
 {key:'not_helpful',label:'Not Helpful',tone:'red',icon:'chat'},
 {key:'reported',label:'Reported',tone:'gold',icon:'flag'}
];

function feedbackEmpty(message='No feedback data available for the selected filters.'){
 return `<div class="feedback-empty">${esc(message)}</div>`;
}

function feedbackBars(rows=[],emptyMessage){
 const safe=Array.isArray(rows)?rows.filter(row=>num(row.count)>0):[];
 if(!safe.length)return feedbackEmpty(emptyMessage);
 const max=Math.max(1,...safe.map(row=>num(row.count)));
 return `<div class="feedback-bars">${safe.map((row,index)=>`<div class="feedback-bar-row"><span title="${esc(row.label||'Uncategorized')}">${esc(row.label||'Uncategorized')}</span><div class="chart-track"><i style="width:${pct(row.count,max)}%;background:${['#1765f5','#22b477','#f5b91c','#8d5ae6','#ea5267','#46bdd1'][index%6]}"></i></div><strong>${num(row.count).toLocaleString()}</strong></div>`).join('')}</div>`;
}

function feedbackDate(value){
 if(!value)return 'Date not recorded';
 const parsed=new Date(value);
 if(Number.isNaN(parsed.getTime()))return String(value).slice(0,19).replace('T',' ');
 return new Intl.DateTimeFormat(undefined,{month:'short',day:'numeric',year:'numeric',hour:'numeric',minute:'2-digit'}).format(parsed);
}

function feedbackBadge(row,U){
 if(row.reported)return U.badge('Reported','gold');
 return U.badge(row.rating==='yes'?'Helpful':'Not Helpful',row.rating==='yes'?'green':'red');
}

const feedbackPage=AdminPages.register('feedback','User Feedback','View and analyze user feedback to understand satisfaction and improve LaborLens.',async(search,page)=>{
 const U=AdminUI,form=page.querySelector('form'),values=new FormData(form);
 const params=new URLSearchParams(values);
 params.set('q',search);
 params.set('offset',page.dataset.offset||0);
 params.set('limit',adminPageSize());
 const offset=Number(page.dataset.offset||0);
 const insightParams=new URLSearchParams({days:values.get('days')||'30',category:values.get('category')||'',language:values.get('language')||''});
 if(values.get('date_from'))insightParams.set('date_from',values.get('date_from'));
 if(values.get('date_to'))insightParams.set('date_to',values.get('date_to'));
 const [data,insights]=await Promise.all([
  AdminPages.api('/admin/feedback-page?'+params),
  AdminPages.api('/admin/insights?'+insightParams)
 ]);
 const feedback=Array.isArray(insights.feedback)?insights.feedback:[];
 const count=key=>num(feedback.find(row=>row.label===key)?.count);
 const distribution=feedbackKinds.map(kind=>({label:kind.label,count:count(kind.key)}));
 const total=distribution.reduce((sum,row)=>sum+num(row.count),0);
 const trend=Array.isArray(insights.feedback_trend)?insights.feedback_trend:[];
 const categoryRows=Array.isArray(insights.feedback_categories)?insights.feedback_categories:[];
 const topicRows=(Array.isArray(insights.feedback_topics)?insights.feedback_topics:[]).slice(0,12);
 const languageRows=(Array.isArray(insights.feedback_languages)?insights.feedback_languages:[]).map(row=>({...row,label:languageLabel(row.label)}));
 const issueRows=Array.isArray(insights.feedback_issues)?insights.feedback_issues:[];
 const cardHtml=`<section class="feedback-summary-cards" aria-label="Feedback summary">${feedbackKinds.map(kind=>`<article class="feedback-summary-card ${kind.tone}"><span>${U.icon(kind.icon)}</span><div><small>${kind.label}</small><strong>${count(kind.key).toLocaleString()}</strong></div></article>`).join('')}<article class="feedback-summary-card blue"><span>${U.icon('chat')}</span><div><small>Total Feedback</small><strong>${total.toLocaleString()}</strong></div></article></section>`;
 const trendSeries=feedbackKinds.map(kind=>({label:kind.label,rows:trend.filter(row=>row.label===kind.key)}));
 const analytics=`<div class="feedback-analytics-grid">
  <section class="panel feedback-panel"><div class="panel-hdr"><h2 class="panel-title">Feedback Distribution</h2></div><div class="panel-body">${total?U.donut(distribution):feedbackEmpty()}</div></section>
  <section class="panel feedback-panel"><div class="panel-hdr"><h2 class="panel-title">Feedback Trend</h2></div><div class="panel-body">${trend.length?U.line(trendSeries):feedbackEmpty()}</div></section>
  <section class="panel feedback-panel"><div class="panel-hdr"><h2 class="panel-title">Feedback by Category</h2></div><div class="panel-body">${feedbackBars(categoryRows)}</div></section>
  <section class="panel feedback-panel"><div class="panel-hdr"><h2 class="panel-title">Feedback by Topic</h2></div><div class="panel-body"><p class="feedback-panel-note">User-submitted feedback grouped by the question’s topic for the selected date range, category, and language.</p>${feedbackBars(topicRows)}</div></section>
  <section class="panel feedback-panel"><div class="panel-hdr"><h2 class="panel-title">Feedback by Language</h2></div><div class="panel-body">${feedbackBars(languageRows)}</div></section>
  <section class="panel feedback-panel"><div class="panel-hdr"><h2 class="panel-title">Top Report / Negative Feedback Comments</h2></div><div class="panel-body">${feedbackBars(issueRows,'No negative feedback comments for this period.')}</div></section>
 </div>`;
 const rows=(Array.isArray(data.items)?data.items:[]).map(row=>[
  esc(feedbackDate(row.created_at)),esc(row.anonymous_user||'Guest'),`<span class="feedback-question">${esc(row.question||'Question not recorded')}</span>`,esc(row.category||'Uncategorized'),esc(languageLabel(row.language)),feedbackBadge(row,U),`<button class="btn btn-sm" data-feedback="${Number(row.id)}">View</button>`
 ]);
 const records=`<section class="feedback-records-card"><div class="feedback-records-title"><h2>Feedback Records</h2><span>${data.total?`Showing ${offset+1}–${Math.min(offset+adminPageSize(),num(data.total))} of ${num(data.total)}`:'No feedback records match your filters.'}</span></div>${U.table(['Date & Time','User','Question','Category','Language','Feedback','Action'],rows,'feedback-records-table')}<div class="workbench-pager" id="feedback-pagination"></div></section>`;
 return {html:cardHtml+analytics+records,mount(host){
  U.pager(host.querySelector('#feedback-pagination'),offset,num(data.total),next=>{page.dataset.offset=String(next);AdminPages.open('feedback');},adminPageSize());
  host.querySelectorAll('[data-feedback]').forEach(button=>button.onclick=()=>{
   const row=(data.items||[]).find(item=>Number(item.id)===Number(button.dataset.feedback));
   if(!row)return;
   U.go('reviews');
   AdminReviews.open(Number(row.message_id));
  });
 }};
});

const feedbackFilters=feedbackPage.querySelector('form');
const feedbackSubmit=feedbackFilters.querySelector('button');
feedbackFilters.classList.add('feedback-filter-panel');
feedbackFilters.querySelector('input[type=search]').closest('label').remove();
feedbackSubmit.innerHTML=`${AdminUI.icon('search')} Apply Filters`;
feedbackSubmit.classList.add('feedback-refresh');
const feedbackDefaultDays=adminReportPeriod(),feedbackDefaultRange=analyticsPresetRange(feedbackDefaultDays);
feedbackSubmit.insertAdjacentHTML('beforebegin',`<label>Category<select name="category"><option value="">All categories</option>${KB_CATEGORIES.map(category=>`<option>${esc(category)}</option>`).join('')}</select></label><label>Language<select name="language"><option value="">All languages</option><option value="en">English</option><option value="fil">Filipino</option><option value="hil">Hiligaynon</option></select></label><label>Feedback<select name="kind"><option value="">All feedback</option><option value="helpful">Helpful</option><option value="not_helpful">Not Helpful</option><option value="reported">Reported</option></select></label><label>Quick range<select name="days"><option value="30" ${feedbackDefaultDays==='30'?'selected':''}>Last 30 days</option><option value="7" ${feedbackDefaultDays==='7'?'selected':''}>Last 7 days</option></select></label><label>From<input name="date_from" type="date" required value="${feedbackDefaultRange.from}"/></label><label>To<input name="date_to" type="date" required value="${feedbackDefaultRange.to}"/></label>`);
feedbackFilters.elements.days.onchange=event=>{const range=analyticsPresetRange(event.target.value);feedbackFilters.elements.date_from.value=range.from;feedbackFilters.elements.date_to.value=range.to;};
feedbackFilters.onreset=()=>setTimeout(()=>{feedbackFilters.elements.days.value=feedbackDefaultDays;feedbackFilters.elements.date_from.value=feedbackDefaultRange.from;feedbackFilters.elements.date_to.value=feedbackDefaultRange.to;feedbackPage.dataset.offset='0';AdminPages.open('feedback');},0);
