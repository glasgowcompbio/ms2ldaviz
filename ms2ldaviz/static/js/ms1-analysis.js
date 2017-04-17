$(document).ready( function(){

	$('#MasterSelectBox').pairMaster();

	$('#btnGroup1Add').click(function(){
		$('#MasterSelectBox').addSelected('#id_group1');
	});

	$('#btnGroup1Remove').click(function(){
		$('#id_group1').removeSelected('#MasterSelectBox');
	});

	$('#btnGroup2Add').click(function(){
		$('#MasterSelectBox').addSelected('#id_group2');
	});

	$('#btnGroup2Remove').click(function(){
		$('#id_group2').removeSelected('#MasterSelectBox');
	});

});

