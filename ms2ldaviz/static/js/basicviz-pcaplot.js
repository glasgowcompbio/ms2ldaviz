function pca_plot(url) {
		var plot_width=950
	var plot_height=950
	var hor_margin = 50
	var ver_margin = 50

	
	d3.json(url,function(error,dataset) {
        if (error) throw error;
        pca_plot(dataset)
        $('#message').fadeOut('fast');
    });

	function pca_plot(pca_data) {
		var def_circ_radius = 4;
		var circ_size = def_circ_radius;
		var def_tip_font_size = 72;
		var tip_font_size = def_tip_font_size;
		var tip_offset_x = 10;
		var tip_offset_y = 10;
		var def_stroke_width = 2.0;
	// var pca_data = [[1,-1,'urine1','#aabbcc'],[3,4,'beer1','#aa0000'],[-3,6,'urine2','#aabbcc']]
		var pca_plot_svg = d3.select("#pca_div")
	               .append("svg")
	                    .attr("width",plot_width)
	                    .attr("height",plot_height)
	                    .attr("id","pca_plot_svg")
                    .append("g")
	                	.call(d3.behavior.zoom().on("zoom",zoom)) 
	                .append("g");

	    pca_plot_svg.append("rect")
    		.attr("class", "overlay")
    		.attr("fill",'#cccccc')
		    .attr("width", plot_width)
		    .attr("height", plot_height);  

	    var circles = pca_plot_svg.selectAll("circle")
	            .data(pca_data[0])
	            .enter()
	            .append("circle")
	            .on("click",function(d) {
	                console.log(d);
	            })
	    var lines = pca_plot_svg.selectAll("line")
	    		.data(pca_data[1])
	    		.enter()
	    		.append("line")

	    // Axis scale objects
	    var max_x = d3.max(pca_data[0],function(d) {return d[0]+1})
	    var min_x = d3.min(pca_data[0],function(d) {return d[0]-1})
	    var max_y = d3.max(pca_data[0],function(d) {return d[1]+1})
	    var min_y = d3.min(pca_data[0],function(d) {return d[1]-1})

	    var xScale = d3.scale.linear()
	    xScale.domain([min_x, max_x])
	    xScale.range([ hor_margin,plot_width-hor_margin])
	    var yScale = d3.scale.linear()
	    yScale.domain([min_y,max_y])
	    yScale.range([plot_height-ver_margin,ver_margin])


	    var xAxis = d3.svg.axis()
	        .scale(xScale)
	        .orient("bottom");

	    pca_plot_svg.append("g")
	        .attr("class", "axis")
	        .attr("transform", "translate(0," + yScale(0) + ")")
	        .attr('stroke-width',def_stroke_width)
	        .attr('id','xax')
	        .call(xAxis);

	    var yAxis = d3.svg.axis()
	        .scale(yScale)
	        .orient("left");

	    pca_plot_svg.append("g")
	    	.attr('id','yax')
	        .attr("class","axis")
	        .attr('stroke-width',def_stroke_width)
	        .attr("transform","translate(" + xScale(0) + ",0)")
	        .call(yAxis);


	    lines.attr("x1",xScale(0))
	    	 .attr("y1",yScale(0))
	    	 .attr("x2", function(d) { return xScale(d[0]);})
	    	 .attr("y2", function(d) { return yScale(d[1]);})
	    	 .attr("stroke-width",def_stroke_width)
	    	 .attr("stroke", function(d) { return d[3];})
	    	 .on("mouseover",function(d) {
    			var xPos = parseFloat(d3.select(this).attr("x2")) - tip_offset_x
	            var yPos = parseFloat(d3.select(this).attr("y2")) - tip_offset_y
	            pca_plot_svg.append("text")
	            	.attr("id","line_tooltip")
	            	.attr('x',xPos)
	            	.attr('y',yPos)
                	.attr("font-family","sans-serif")
                	.attr("font-weight","bold")
	                .attr("font-size",""+ tip_font_size + "px")
                	.text(d[2]);
	    	 })
	    	 .on("mouseout",function(d) {
	    	 	d3.select("#line_tooltip").remove()
	    	 });
	    	 
	    circles.attr("cx",function(d) {return xScale(d[0])})
	    		.attr("cy",function (d) {return yScale(d[1])})
	    		.attr("r",def_circ_radius)
	    		.attr('fill',function(d) {return d[3]})
	    		.on("mouseover",function(d) {
	    			var xPos = parseFloat(d3.select(this).attr("cx")) - tip_offset_x
		            var yPos = parseFloat(d3.select(this).attr("cy")) - tip_offset_y
		            d3.select(this)
	                	.transition()
	                	.duration(250)
	                	.attr("r",2*circ_size)
	                	.attr("fill",'#aaaaff');
	                pca_plot_svg.append("text")
	                	.attr("id","tooltip")
	                	.attr('x',xPos)
	                	.attr('y',yPos)
	                	.attr("font-family","sans-serif")
	                	.attr("font-weight","bold")
		                .attr("font-size",""+ tip_font_size + "px")
	                	.text(d[2]);
	            })
	            .on("mouseout",function(d) {
	            	d3.select(this)
		            	.transition()
		                .duration(250)
	            		.attr("r",circ_size)
	            		.attr('fill',d[3]);
	            	d3.select('#tooltip').remove()
	            });

	    

	    function zoom() {
	    	pca_plot_svg.attr("transform","translate(" + d3.event.translate + ")scale(" + d3.event.scale + ")");
	    	pca_plot_svg.selectAll('circle')
	    		.attr('r',def_circ_radius/d3.event.scale);
	    	circ_size = def_circ_radius/d3.event.scale;
	    	tip_font_size = def_tip_font_size/d3.max([10.0,d3.event.scale]);
	    	tip_offset_x = 20.0/d3.max([10.0,d3.event.scale]);
	    	tip_offset_y = 20.0/d3.max([10.0,d3.event.scale]);
	    	pca_plot_svg.selectAll('line')
	    		.attr('stroke-width',def_stroke_width/d3.event.scale)
	    	pca_plot_svg.select('#yax')
	    		.attr('stroke-width',def_stroke_width/d3.event.scale)
	    	pca_plot_svg.select('#xax')
	    		.attr('stroke-width',def_stroke_width/d3.event.scale)

	    }
    }

}