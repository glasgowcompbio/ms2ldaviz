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
	var margin = {top: 30, right: 10, bottom: 30, left: 150}	

	//data
	var names = []
	var values = []
	
	for (i=0; i < total_dataset[1].length; i++) {		
		names.push(total_dataset[1][i][0]);
		values.push(total_dataset[1][i][1]);		
	}

	var docCount = total_dataset[0]

	//x and y scales
	var xscale = d3.scale.linear()
					.domain([0, docCount]) //sets x values up to total amount of docs
					.range([margin.left, width+margin.left]); //0-n pixels on the canvas

	var yscale = d3.scale.linear()
					.domain([0, names.length]) //sets y values to amount of categs
					.range([margin.top, height+margin.top]); //length of y axis

	var canvas = d3.select("#graphs")					
					.append("svg")
					.attr('id', 'canvas')
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
			.ticks(names.length)
			.tickFormat(function(d,i){ return names[i]; });
			

	//added axis to canvas, plus their positions
	var x_axis = canvas.append('g')
					  .attr("transform", "translate(0," + (height+margin.top) + ")")
					  .attr('id','x_axis')
					  .call(xAxis
					  .tickFormat(d3.format("s")));
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
						.style('fill', function(d, i) {	return total_dataset[1][i][2]; })
						.attr('width', function(d, i) {return xscale(total_dataset[1][i][1])-margin.left})
						.attr('stroke', '#ffffff')
						.on("mouseover",function(d, i) {
				            d3.select(this)
				                // .attr("stroke","green")
				                .attr("stroke-width",2);
			                    var xPos = xscale(d/2);
			            		function yPos(d,i){ return yscale(i); };
				            	canvas.append("text")
				                .attr("id","tooltip")
				                .attr("x",xPos)
				                .attr("y",yPos(d,i))
				                .attr("class", "tooltips")				          
				                .text(+total_dataset[1][i][1].toFixed(2))})
						.on("mouseout",function() {
				            d3.select(this)
				                .transition()
				                .duration(250)
				                .attr("stroke-width",1);
				            d3.select("#tooltip").remove()
				        });
						// .on('mouseover', function(){ return tooltip.style('visibility', 'visible')})						
						// .on('mouseout', function(){ return tooltip.style('visibility', 'hidden')});
	
	// var animation = d3.select("#bars")
	// 					.data(values)
	// 					.enter()
	// 				    .transition()
	// 				    .duration(700) 
	// 				    .attr('width', function(d, i) {return xscale(total_dataset[i][1])-margin.left});



}