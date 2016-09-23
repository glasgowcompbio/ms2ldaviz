function bar_plot(url,div_id) {

	d3.json(url,function(error,dataset) {
	        if (error) throw error;
	        draw_plot(dataset,div_id)
	        $('#message').fadeOut('fast');
	    });

	function draw_plot(dataset,div_id) {

		var hor_margin = 70
		var ver_margin = 30
		var plot_width = 600
		var plot_height = 300

		var bar_plot_svg = d3.select("#" + div_id)
				               .append("svg")
			                    .attr("width",plot_width)
			                    .attr("height",plot_height)
			                    .attr("id","bar_plot_svg")

		// bar_plot_svg.append("rect")
		//     		.attr("class", "overlay")
		//     		.attr("fill",'#cccccc')
		// 		    .attr("width", plot_width)
		// 		    .attr("height", plot_height);  



		var max_alpha = d3.max(dataset,function(d) {return d[1]}) * 1.2
		var n_bars = dataset.length

		// console.log(n_bars)

		var xScale = d3.scale.linear()
	    xScale.domain([0, n_bars])
	    xScale.range([ hor_margin,plot_width-hor_margin])
	    var yScale = d3.scale.linear()
	    yScale.domain([0,max_alpha])
	    yScale.range([plot_height-ver_margin,ver_margin])

	    var rectangles = bar_plot_svg.selectAll("rect")
	    								.data(dataset)
	    								.enter()
	    								.append("rect")

	    var bar_width = 1.0*(plot_width - 2*hor_margin)/n_bars
	    height_factor = (plot_height - 2*ver_margin)/max_alpha


		
	    // var xAxis = d3.svg.axis()
	    //     .scale(xScale)
	    //     .orient("bottom");

	    // bar_plot_svg.append("g")
	    //     .attr("class", "axis")
	    //     .attr("transform", "translate(0," + yScale(0) + ")")
	    //     .attr('stroke-width',1)
	    //     .attr('id','xax')
	    //     .call(xAxis);

	    var yAxis = d3.svg.axis()
	        .scale(yScale)
	        .orient("left");

	    bar_plot_svg.append("g")
	    	.attr('id','yax')
	        .attr("class","axis")
	        .attr('stroke-width',1)
	        .attr("transform","translate(" + xScale(0) + ",0)")
	        .call(yAxis);

	    var a = 0
	    rectangles.attr("x",function(d,i) {return xScale(i)})
	    			.attr("y",function(d) {return yScale(d[1])})
	    			.attr("width",0.9*bar_width)
	    			.attr("height",function(d) { return yScale(0) - yScale(d[1]);})
	    			.attr("stroke",'#000000')
	    			.attr("fill",'#AA0000')
	    			.on("mouseover",function(d,i) {
			            d3.select(this)
		                	.transition()
		                	.duration(250)
		                	.attr("fill",'#aaaaff');
		                bar_plot_svg.append("text")
		                	.attr("id","tooltip")
		                	.attr('x',plot_width/2-100)
		                	.attr('y',50)
		                	.attr("font-family","sans-serif")
		                	.attr("font-weight","bold")
			                .attr("font-size",""+ 14 + "px")
		                	.text(d[0]);
		                scatter_highlight(i);
		            })
		            .on("mouseout",function(d,i) {
		            	d3.select(this)
			            	.transition()
			                .duration(250)
		            		.attr('fill','#AA0000');
		            	d3.select('#tooltip').remove();
		            	scatter_unhighlight(i);
		            });

	}
}