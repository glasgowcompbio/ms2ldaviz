function plot_word_graph(mass2motif_id, motif_name){
	// current_pos = 0

    //clear the existing svg (if it exists)
    d3.select('#canvas').remove()
 
    var url = '/basicviz/get_word_graph/' + mass2motif_id + '/';
    d3.json(url,function(error,total_dataset) {    	
        if (error) throw error;
        plot(mass2motif_id, total_dataset);
    });

function plot(mass2motif_id, total_dataset){
	var height = 350;
	var width = 600;

	var margin = {top: 30, right: 10, bottom: 30, left: 10}

	var x_length = total_dataset[0][1]; //length will always be the first count of dataset

	var values = 300;

	//data
	var names = []
	for (i=0; i < total_dataset.length; i++) {
		if (total_dataset[i][1] > 0){
			names.push(total_dataset[i][0]);
		}
	}

	var values = [];
	for (i=0; i < total_dataset.length; i++) {
		if (total_dataset[i][1] > 0){
			values.push(total_dataset[i][1]);
		}
	}
	var y_length = names.length
	
	//colors for the bars
	var color = ['#0000b4'];

	//setting things up - grid
	var grid = d3.range(16).map(function(i){
		return {'x1':40,'y1':40,'x2':0,'y2':340};
	});
	
	//x and y scales
	var xscale = d3.scale.linear()
					.domain([0,x_length + 5]) //sets x values up to count thing
					.range([0,width]); //0-n pixels on the canvas

	var yscale = d3.scale.linear()
					.domain([0, d3.max(names, function() { return names.length; })]) //sets y values to amount of categs
					.range([40, height	]); //length of y axis



	//new svg element 900x550
	var canvas = d3.select("body")
					.append("svg")
					.attr({'width':width+margin.left+margin.right,'height':height+margin.bottom+margin.top});
					

	//g element is used to group svg elems together so you can change all of them at once
	var grids = canvas.append('g')
					  .attr('id','grid')
					  .attr('transform','translate(149,10)') //position of the grid
					  .selectAll('line')
					  .data(grid)
					  .enter()
					  .append('line')
					  .attr({'x1':function(d,i){ return i*30; },
							 'y1':function(d){ return d.y1; },
							 'x2':function(d,i){ return i*30; },
							 'y2':function(d){ return d.y2; },
						})
					  .style({'stroke':'#adadad','stroke-width':'1px'});

	//x axis added to svg, oriented bottom and added to scale 
	var	xAxis = d3.svg.axis();
	xAxis
		.orient('bottom')
		.scale(xscale)
		.tickSize(3)
		.tickValues(d3.range(20));

	var	yAxis = d3.svg.axis();
		yAxis
			.orient('left')
			.scale(yscale)
			.tickSize(3)
			.tickFormat(function(d,i){ return names[i]; })
			.tickValues(d3.range(15));

	//added axis to canvas, plus their positions
	var y_axis = canvas.append('g')
					  .attr("transform", "translate(150,0)")
					  .attr('id','y_axis')
					  .call(yAxis);

	var x_axis = canvas.append('g')
					  .attr("transform", "translate(150, 350)")
					  .attr('id','x_axis')
					  .call(xAxis);

	//adding data to the rectangles
	var chart = canvas.append('g')
						.attr("transform", "translate(150,0)") //position of the bars
						.attr('id','bars')
						.selectAll('rect')
						.data(values)
						.enter()
						.append('rect')
						.attr('height',20)
						.attr({'x':0,'y':function(d,i){ return yscale(i)+19; }}) //text position+text
						.style('fill', color)
						.attr('width',function(d){ return 20; });

	//animation
	var animation = d3.select("svg").selectAll("rect")
					    .data(values)
					    .transition()
					    .duration(700) 
					    .attr("width", function(d) {return xscale(d); });

	var textonbars = d3.select('#bars')
						.selectAll('text')
						.data(values)
						.enter()
						.append('text')
						.attr({'x':function(d) {return xscale(d)-20; },'y':function(d,i){ return yscale(i)+35; }})
						.text(function(d){ return d; }).style({'fill':'#fff','font-size':'14px'});
};
}