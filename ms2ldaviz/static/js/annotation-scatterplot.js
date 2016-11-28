function scatter_plot(dataset) {
	var plot_width = 500;
	var plot_height = 500;
	var hor_margin = 50;
	var ver_margin = 50;
	var def_stroke_width = 0.5;
	var def_circ_radius = 5.0;
	var tip_offset_x = 10.0;
	var tip_offset_y = 15.0;

	var scatter_plot_svg = d3.select('#scatter_div')
		.append("svg")
			.attr("width",plot_width)
			.attr("height",plot_height)
			.attr("id","scatter_plot_svg");

	scatter_plot_svg.append("rect")
		.attr("class", "overlay")
    		.attr("fill",'#cccccc')
		    .attr("width", plot_width)
		    .attr("height", plot_height);

	var circles = scatter_plot_svg.selectAll("circle")
	            .data(dataset)
	            .enter()
		            .append("circle")
		            .on("click",function(d) {
		                console.log(d);
		            });
	var max_x = 1.0;
    var min_x = 0.0;
    var max_y = 1.0;
    var min_y = 0.0;

    var xScale = d3.scale.linear();
    xScale.domain([min_x, max_x]);
    xScale.range([ hor_margin,plot_width-hor_margin]);
    var yScale = d3.scale.linear();
    yScale.domain([min_y,max_y]);
    yScale.range([plot_height-ver_margin,ver_margin]);

    var xAxis = d3.svg.axis()
        .scale(xScale)
        .orient("bottom");

    scatter_plot_svg.append("g")
        .attr("class", "axis")
        .attr("transform", "translate(0," + yScale(0) + ")")
        .attr('id','xax')
        .call(xAxis);

    var yAxis = d3.svg.axis()
        .scale(yScale)
        .orient("left");

    scatter_plot_svg.append("g")
    	.attr('id','yax')
        .attr("class","axis")
        // .attr('stroke-width',def_stroke_width)
        .attr("transform","translate(" + "50" + ",0)")
        .call(yAxis);

	circles.attr("cx",function(d) {return xScale(d[0])})
		    		.attr("cy",function (d) {return yScale(d[1])})
		    		.attr("r",def_circ_radius)
		    		.attr('fill',function(d) {return 'red'})
		    		.attr('stroke','black')
		    		.on("mouseover",function(d) {
		    			var xPos = parseFloat(d3.select(this).attr("cx")) - tip_offset_x
			            var yPos = parseFloat(d3.select(this).attr("cy")) - tip_offset_y
			            d3.select(this)
		                	.transition()
		                	.duration(250)
		                	.attr("r",2*def_circ_radius)
		                	.attr("fill",'#aaaaff');
		                scatter_plot_svg.append("text")
		                	.attr("id","tooltip")
		                	.attr('x',xPos)
		                	.attr('y',yPos)
		                	.attr("font-family","sans-serif")
		                	.attr("font-weight","bold")
			                .attr("font-size",""+ 14 + "px")
		                	.text(d[2]);
		            })
		            .on("mouseout",function(d) {
		            	d3.select(this)
			            	.transition()
			                .duration(250)
		            		.attr("r",def_circ_radius)
		            		.attr('fill','red');
		            	d3.select('#tooltip').remove()
	            	});

	var xAxisLabel = scatter_plot_svg.append('text')
						.text('Probability')
						.attr('x',xScale(0.4))
						.attr('y',yScale(-0.1));
	var yAxisLabel = scatter_plot_svg.append('text')
						.text('Overlap')
						.attr('x',xScale(-0.1))
						.attr('y',yScale(0.5))
						.attr("transform","rotate(-90,"+xScale(-0.07) + "," + yScale(0.5) + ")");
}