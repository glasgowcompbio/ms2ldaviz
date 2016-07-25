function pca_plot(experiment_id) {
		var plot_width=800
	var plot_height=500
	var hor_margin = 50
	var ver_margin = 50

	
	d3.json("/basicviz/get_pca_data/"+ experiment_id + "/",function(error,dataset) {
        if (error) throw error;
        pca_plot(dataset)
        $('#message').fadeOut('fast');
    });

	function pca_plot(pca_data) {
		var def_circ_radius = 4;
		var circ_size = def_circ_radius;
		var def_tip_font_size = 48;
		var tip_font_size = def_tip_font_size;
		var tip_offset_x = 10;
		var tip_offset_y = 10;
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
	            .data(pca_data)
	            .enter()
	            .append("circle")
	            .on("click",function(d) {
	                console.log(d);
	            })
	    // Axis scale objects
	    var max_x = d3.max(pca_data,function(d) {return d[0]+1})
	    var min_x = d3.min(pca_data,function(d) {return d[0]-1})
	    var max_y = d3.max(pca_data,function(d) {return d[1]+1})
	    var min_y = d3.min(pca_data,function(d) {return d[1]-1})

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
	        .call(xAxis);

	    var yAxis = d3.svg.axis()
	        .scale(yScale)
	        .orient("left");

	    pca_plot_svg.append("g")
	        .attr("class","axis")
	        .attr("transform","translate(" + xScale(0) + ",0)")
	        .call(yAxis);

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
	    }
    }

}