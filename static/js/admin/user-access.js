async function setUserAccess(id,active){
  if(!confirm(`${active?'Reactivate':'Disable'} this user account? Existing conversations will be retained.`))return false;
  try{await AdminPages.api(`/admin/users/${id}/access`,{method:'PUT',body:JSON.stringify({is_active:active})});toast('Account access updated.');await loadUsers();return true;}
  catch(e){toast(e.message,true);return false;}
}
