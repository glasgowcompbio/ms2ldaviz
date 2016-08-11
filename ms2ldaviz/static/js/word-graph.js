function plot_word_graph(url, mass2motif_id, motif_name){
	// current_pos = 0

    //clear the existing svg (if it exists)
    d3.select('#canvas').remove()
 
    
    d3.json(url,function(error,total_dataset) {    	
        if (error) throw error;
        plot(mass2motif_id, total_dataset);
    });
}
function plot(mass2motif_id, total_dataset){
	var height = 350;
	var width = 600;
	var margin = {top: 30, right: 10, bottom: 30, left: 200}
	

	//data
	var names = []
	var values = []
	var docCount = 0
	for (i=0; i < total_dataset.length; i++) {
		if (total_dataset[i][1] > 0){
			names.push(total_dataset[i][0]);
			values.push(total_dataset[i][1]);
		}
		docCount = i;
	}
	if (docCount < total_dataset[0][1])	
		docCount = total_dataset[0][1];

	//x and y scales
	var xscale = d3.scale.linear()
					.domain([0, docCount]) //sets x values up to total amount of docs
					.range([margin.left, width+margin.left]); //0-n pixels on the canvas

	var yscale = d3.scale.linear()
					.domain([0, names.length]) //sets y values to amount of categs
					.range([margin.top, height+margin.top]); //length of y axis


	d3.select('#canvas').remove()
	//new svg element 900x550
	var canvas = d3.select("body")
					.append("graphs")
					.append("svg")
					.attr({'width':width+margin.left+margin.right,'height':height+margin.bottom+margin.top});
					


	//x axis added to svg, oriented bottom and added to scale 
	var	xAxis = d3.svg.axis();
	xAxis
		.orient('bottom')
		.scale(xscale);

	var	yAxis = d3.svg.axis();
		yAxis
			.orient('left')
			.scale(yscale)
			.tickFormat(function(d,i){ return names[i]; });
			

	//added axis to canvas, plus their positions
	var x_axis = canvas.append('g')
					  .attr("transform", "translate(0," + (height+margin.top) + ")")
					  .attr('id','x_axis')
					  .call(xAxis);
	var y_axis = canvas.append('g')
					  .attr("transform", "translate(" + (margin.left) + ",0)")
					  .attr('id','y_axis')
					  .call(yAxis);

	//adding data to the rectangles
	var barHeight = (height/names.length)
	var chart = canvas.append('g') 
						.attr('id','bars')
						.selectAll('rect')
						.data(values)
						.enter()
						.append('rect')
						.attr('height', barHeight)
						.attr({'x':xscale(0),'y':function(d,i){ return yscale(i); }}) //bar pos
						.style('fill', function(d, i) {
							var color = total_dataset[i][2];							
							return color;})
						.attr('width', 0)
						.attr('stroke', '#ffffff');

	animation
	var animation = d3.select("svg").selectAll("rect")
					    .data(values)
					    .transition()
					    .duration(700) 
					    .attr('width', function(d, i) {return xscale(total_dataset[i][1])-margin.left});

	var textonbars = d3.select('#bars')
						.selectAll('text')
						.data(values)
						.enter()
						.append('text')
						.attr({'x':function(d) {return xscale(d)-15; },'y':function(d,i){ return yscale(i)+18; }})
						.text(function(d){ return d; })
						.style({'fill':'#fff','font-size':'14px'});

}