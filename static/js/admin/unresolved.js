/* -- Unresolved Queries -- */
async function loadUnresolved() {
  const tb=document.getElementById('queries-tbody');
  const label=document.getElementById('unresolved-label');
  try {
    const r=await apiFetch('/admin/unresolved-queries');
    const qs=await r.json();
    if(!r.ok) throw new Error(qs.detail||'Failed');
    const badge=document.getElementById('unresolved-nav-badge');
    if(!qs.length){
      label.textContent='0 pending';
      badge.style.display='none';
      tb.innerHTML='<tr><td colspan="8" class="empty" style="color:var(--green);font-weight:600;">All questions matched results in the reference documents.</td></tr>';
      return;
    }
    label.textContent=`${qs.length} pending review`;
    badge.textContent=qs.length; badge.style.display='inline-flex';
    tb.innerHTML=qs.map((q,i)=>`<tr class="unresolved-row">
      <td class="td-mono">${i+1}</td>
      <td class="td-strong">${esc(q.anonymous_user||'Guest')}</td>
      <td style="max-width:280px;"><div class="no-kb-tag">NO SOURCE MATCH</div><div style="font-size:13px;margin-top:4px;">${esc(q.query_text)}</div></td>
      <td><span class="badge badge-amber">${esc((q.intent||'').replace(/_/g,' ').toUpperCase())}</span></td>
      <td style="max-width:250px;font-size:12px;color:var(--muted);">${esc(unresolvedSuggestionText(q.intent))}</td>
      <td class="td-mono">${q.response_time_ms??0}ms</td>
      <td class="td-mono">${dateLabel(q.created_at)}</td>
      <td><div style="display:flex;gap:6px;flex-wrap:wrap;"><button class="btn btn-sm btn-success" onclick="resolveQ(${q.id})">Mark Reviewed</button><button class="btn btn-sm btn-gold" onclick="addToFAQ('${jsArg(q.query_text)}')">Add FAQ</button></div></td>
    </tr>`).join('');
  } catch(e){tb.innerHTML=`<tr><td colspan="8"><div class="err-msg">${esc(e.message)}</div></td></tr>`;}
}

async function resolveQ(id) {
  try {
    const r=await apiFetch(`/admin/unresolved-queries/${id}/resolve`,{method:'PUT'});
    if(!r.ok) throw new Error('Failed.');
    toast('Resolved.'); loadUnresolved(); loadOverview();
  } catch(e){toast(e.message,true);}
}

function addToFAQ(q) {
  editingFaqId = null;
  document.getElementById('faq-q').value=q;
  document.getElementById('faq-a').value='';
  document.querySelectorAll('.nav-item').forEach(n=>n.classList.remove('active'));
  document.querySelectorAll('.content-inner > [id^="tab-"]').forEach(t=>t.classList.remove('tab-active'));
  document.querySelector('[data-tab="faqs"]').classList.add('active');
  document.getElementById('tab-faqs').classList.add('tab-active');
  document.getElementById('faq-a').focus();
  toast('Query copied to FAQ form. Write an answer and publish it.');
}

