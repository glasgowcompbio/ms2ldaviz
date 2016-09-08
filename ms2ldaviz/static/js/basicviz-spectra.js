function load_parents(mass2motif_id,motif_name,vo_id) {

    current_pos = 0

    //clear the existing svg (if it exists)
    d3.select('#load_text').remove()
    d3.select("#frag_graph_titlebar_svg").remove();
    d3.select("#frag_graph_svg").remove();
    d3.select('#frag_graph_svg').remove()

    if(vo_id > -1) {
        var url = '/basicviz/get_parents/' + mass2motif_id + '/' + vo_id + '/';
    }else {
        var url = '/basicviz/get_parents/' + mass2motif_id + '/';
    }

    d3.json(url,function(error,total_dataset) {
        if (error) throw error;
        key = null
        plot_parent(total_dataset,motif_name)
    });

}

function load_parent(url) {
    $('#status').text('Loading motif')
    current_pos = 0
    d3.json(url,function(error,total_dataset) {
    if (error) throw error;
    key = total_dataset[0][1]
    console.log(key)
    plot_parent(total_dataset[0][0],"")
    });
    $('#status').text('Loaded')
}

function plot_parent(total_dataset,motif_name) {

    if(current_pos < 0) {
        current_pos = 0
    }
    if(current_pos > total_dataset.length-1) {
        current_pos = total_dataset.length-1
    }

    var parent_mass = total_dataset[current_pos][0][0]
    var parent_intensity = total_dataset[current_pos][0][1]
    var parent_name = total_dataset[current_pos][0][2]
    var parent_annotation = total_dataset[current_pos][0][3]
    var dataset = total_dataset[current_pos][1]
    var max_mass = d3.max(dataset,function(d) {return d[0]+50})

    if (max_mass < parent_mass) {
        max_mass = parent_mass + 50;
    }

    var total_width = 900
    var total_height = 500
    var plot_width = 700
    var plot_height = 300
    var ver_margin = 30
    var hor_margin = 50
    var head_height = 50

    var n_parents = total_dataset.length
    n_parents -= 1

    //clear the existing svg (if it exists)
    d3.select('#load_text').remove()
    d3.select("#frag_graph_titlebar_svg").remove();
    d3.select("#frag_graph_svg").remove();

    var frag_graph_titlebar_svg = d3.select("#spectra")
       .append("svg")
            .attr("width",plot_width)
            .attr("height",head_height)
            .attr("id","frag_graph_titlebar_svg")
    frag_graph_titlebar_svg.append("rect")
        .attr("width","98%")
        .attr("height","98%")
        .attr("x","10")
        .attr("stroke","black")
        .attr("fill","#F2F2F2");


    if (key != null) {
        var keybox = d3.select("#spectra")
            .append("svg")
                .attr("width",plot_width)
                .attr("height",140)
                .attr("id","key_box_svg")
                .attr("x",0)
                .attr("y",360)
        keybox.append("rect")
            .attr("width","98%")
            .attr("height","98%")
            .attr("x","10")
            .attr("stroke","black")
            .attr("fill","#F2F2F2");
        keylines = keybox.selectAll("lines")
                    .data(key)
                    .enter()
                    .append("line");
        keylines.attr("x1",50)
                 .attr("x2",300)
                 .attr("y1",function(d,i) {return 20 + 20*i;})
                 .attr("y2",function(d,i) {return 20 + 20*i;})
                 .attr("stroke",function(d) {return d[1];})
                 .attr("stroke-width",5);
        keytext = keybox.selectAll("text")
            .data(key)
            .enter()
            .append("text");
        keytext.attr("x",310)
                 .attr("y",function(d,i) {return 20 + 20*i;})
                .text(function(d) {return d[0];});
    };
    var nextbut = frag_graph_titlebar_svg.append("g")
        .attr("cursor","pointer")
        .on("click",function () {current_pos += 1;plot_parent(total_dataset,motif_name)})

    nextbut.append("rect")
                .attr("height",30)
                .attr("width",70)
                .attr("x",plot_width-100)
                .attr("y",9)
                .attr("stroke","black")
                .attr("fill","#CCAAAA")
    nextbut.append("text")
                .attr("x",plot_width-90)
                .attr("y",30)
                .text("Next")
                .attr("font-family","sans-serif")
                .attr("font-size","14px")

    var prevbut = frag_graph_titlebar_svg.append("g")
        .attr("cursor","pointer")
        .on("click",function () {current_pos -= 1;plot_parent(total_dataset,motif_name)})

    prevbut.append("rect")
                .attr("height",30)
                .attr("width",70)
                .attr("x",30)
                .attr("y",9)
                .attr("stroke","black")
                .attr("fill","#CCAAAA")
    prevbut.append("text")
                .attr("x",40)
                .attr("y",30)
                .text("Previous")
                .attr("font-family","sans-serif")
                .attr("font-size","14px")

    frag_graph_titlebar_svg.append("text")
                .attr("x",plot_width-500)
                .attr("y",40)
                .attr("font-family","sans-serif")
                .attr("font-size","14px")
                .text("Parent: " + parent_name + ", " + parent_annotation + "  (" + (current_pos+1) + "/" + (n_parents+1) + ")")
    frag_graph_titlebar_svg.append("text")
                .attr("x",plot_width-500)
                .attr("y",20)
                .text(motif_name)

    var frag_graph_svg = d3.select("#spectra")
               .append("svg")
                    .attr("width",plot_width)
                    .attr("height",plot_height)
                    .attr("y",head_height)
                    .attr("id","frag_graph_svg")

    // Create the line objects
    lines = frag_graph_svg.selectAll("line")
            .data(dataset)
            .enter()
            .append("line")
            .on("click",function(d) {
                console.log(d);
            })

    // Axis scale objects
    var xScale = d3.scale.linear()
    xScale.domain([0, max_mass])
    xScale.range([ hor_margin,plot_width-hor_margin])
    var yScale = d3.scale.linear()
    yScale.domain([0,d3.max(dataset,function(d) {return d[3];})])
    yScale.range([plot_height-ver_margin,ver_margin])

    // Set the line attributes
    lines.attr("x1",function(d) {return xScale(d[0])})
          .attr("x2",function(d) {return xScale(d[1])})
          .attr("y1",function(d) {return yScale(d[2])})
          .attr("y2",function(d) {return yScale(d[3])})
          .attr("stroke",function(d) {return d[5]})
          .attr("stroke-width",2)
        .attr("stroke-dasharray",function(d) {if(d[4]==1) {return "0.0";} else {return "5.5";}})
        .on("mouseover",function(d) {

            d3.select(this)
                // .attr("stroke","green")
                .attr("stroke-width",4);
                    var xPos = parseFloat(d3.select(this).attr("x1"))
            var yPos = parseFloat(d3.select(this).attr("y2"))

            frag_graph_svg.append("text")
                .attr("id","tooltip")
                .attr("x",xPos-50)
                .attr("y",yPos-5)
                .attr("font-family","sans-serif")
                .attr("font-size","12px")
                .attr("font-weight","bold")
                .text(d[6])

        })
        .on("mouseout",function() {
            d3.select(this)
                .transition()
                .duration(250)
                // .attr("stroke",d[5])
                .attr("stroke-width",2);
            d3.select("#tooltip").remove()
            // d3.select("#lossline").remove()
            // d3.select("#losstext").remove()
        });

     // Axes
    var xAxis = d3.svg.axis()
        .scale(xScale)
        .orient("bottom");
    frag_graph_svg.append("g")
        .attr("class", "axis")
        .attr("transform", "translate(0," + (plot_height-ver_margin) + ")")
        .call(xAxis);

    var yAxis = d3.svg.axis()
        .scale(yScale)
        .orient("left")
    frag_graph_svg.append("g")
        .attr("class","axis")
        .attr("transform","translate(" + hor_margin + ",0)")
        .call(yAxis)

    // Axes labels
    frag_graph_svg.append("text")
        .text("Mass")
        .attr("x",plot_width/2)
        .attr("y",plot_height)
        .attr("class","axis-label")

    frag_graph_svg.append("text")
        .text("Relative Intensity")
        .attr("x",0)
        .attr("y",0)
        .attr("class","axis-label")
        .attr("transform","translate(10," + (30+plot_height/2) + ")rotate(-90)");

    frag_graph_svg.append("line")
        .attr("x1",xScale(parent_mass))
        .attr("x2",xScale(parent_mass))
        .attr("y1",plot_height-ver_margin)
        .attr("y2",yScale(parent_intensity))
        .attr("stroke","blue")
        .attr("stroke-width",3)

    frag_graph_svg.append("text")
        .text("Parent Ion (" + parent_mass.toFixed(4) + ")")
        .attr("class","axis-label")
        .attr("transform","translate("+ (xScale(parent_mass)-5) + "," + (50+plot_height/2) + ")rotate(-90)")

};