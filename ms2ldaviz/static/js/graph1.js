//size of the bar chart



// For a particular m2m:
// 		Find all features in that m2m (Mass2MotifInstance) (possibly include probability threshold)
// 		set all feature counts to zero
// 		Find all documents that use that m2m (DocumentMass2Motif)
// For each document:
// 		Find all feature instances in the document (FeatureInstance)
// 		For each feature instance:
// 		If the feature is in the m2m (see list made earlier) 
//			then extract FeatureMass2MotifInstance from FeatureMass2MotifInstance
// 			multiply the featuremass2motifinstance.probability by the featureinstance intensity 
// 			increase feature count for this feature by the result
function plot_this_graph(mass2motif_id, motif_name, annotated) {

	current_pos = 0

    //clear the existing svg (if it exists)
    d3.select('#load_text').remove()
    d3.select("#frag_graph_titlebar_svg").remove();
    d3.select("#frag_graph_svg").remove();
    d3.select('#frag_graph_svg').remove()
    var url = '/basicviz/get_features/' + mass2motif_id + '/';
    d3.json(url,function(error,total_dataset) {
        if (error) throw error;
        do_all_the_work(total_dataset,motif_name)
        $('#message').fadeOut('fast');
    });
}


function do_all_the_work(total_dataset, motif_name) {

		var height = 200;
		var width = 600;

		var margin = {top: 30, right: 10, bottom: 30, left: 10}

		var values = 300;

		//data
		var categories= ['a', 'f', 'r'];
		var dollars = [213,209,190];

		//colors for the bars
		var color = ['#0000b4'];

		//setting things up - grid
		var grid = d3.range(16).map(function(i){
			return {'x1':40,'y1':40,'x2':0,'y2':180};
		});
		
		//x and y scales
		var xscale = d3.scale.linear()
						.domain([0,values]) //sets x values up to 300
						.range([0,width]); //0-n pixels on the canvas

		var yscale = d3.scale.linear()
						.domain([0,categories.length]) //sets y values to amount of categs
						.range([40,height]); //length of y axis


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
			.tickSize(3);

		var	yAxis = d3.svg.axis();
			yAxis
				.orient('left')
				.scale(yscale)
				.tickSize(3)
				.tickFormat(function(d,i){ return categories[i]; })
				.tickValues(d3.range(17));

		//added axis to canvas, plus their positions
		var y_axis = canvas.append('g')
						  .attr("transform", "translate(150,0)")
						  .attr('id','y_axis')
						  .call(yAxis);

		var x_axis = canvas.append('g')
						  .attr("transform", "translate(150,200)")
						  .attr('id','x_axis')
						  .call(xAxis);

		//adding data to the rectangles
		var chart = canvas.append('g')
							.attr("transform", "translate(150,0)") //position of the bars
							.attr('id','bars')
							.selectAll('rect')
							.data(dollars)
							.enter()
							.append('rect')
							.attr('height',19)
							.attr({'x':0,'y':function(d,i){ return yscale(i)+19; }}) //text position+text
							.style('fill', color)
							.attr('width',function(d){ return 20; });

		//animation
		var animation = d3.select("svg").selectAll("rect")
						    .data(dollars)
						    .transition()
						    .duration(700) 
						    .attr("width", function(d) {return xscale(d); });

		var textonbars = d3.select('#bars')
							.selectAll('text')
							.data(dollars)
							.enter()
							.append('text')
							.attr({'x':function(d) {return xscale(d)-100; },'y':function(d,i){ return yscale(i)+35; }})
							.text(function(d){ return d; }).style({'fill':'#fff','font-size':'14px'});
};
