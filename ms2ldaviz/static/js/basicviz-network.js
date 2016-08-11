// see:
// - http://bl.ocks.org/mbostock/3750558
// - https://github.com/mbostock/d3/wiki/Force-Layout
// - http://www.coppelia.io/2014/07/an-a-to-z-of-extra-features-for-the-d3-force-layout/

function plot_graph(vo_id) {

    Math.seedrandom('hello');

    var width = window.innerWidth;
    var height = window.innerHeight;
    var topicName = 'motif';

    var minNodeSize = 8;
    var topicTextSize = 48;
    var docTextSize = 12;
    var size = d3.scale.pow().exponent(1)
        .domain([1, 100])
        .range([8, 24]);

    var defaultNodeColour = '#CCCCCC';
    var specialNodeColour = '#CC0000';
    var minScore = 0;
    var maxScore = 1;
    var color = d3.scale.linear()
        .domain([minScore, (minScore + maxScore) / 2, maxScore])
        .range(['#1f77b4', '#2ca02c', '#ff7f0e']);

    var simulationNumber = 10;
    var simulationTimeout = 1;
    var optArray = [];
    var toggle = 0;
    var url = '/basicviz/get_graph/' + vo_id
    d3.json(url, function(error, graph) {

        if (error) throw error;

        var force = d3.layout.force()
            .size([width, height])
            .charge(-10000)
            // .linkDistance(function(d) {
            //     return d.weight*100;
            // })
            .friction(0.40)
            .on('tick', tick);

        var drag = force.drag()
            .on('dragstart', dragstart);

        // the initial zoom and transform on the svg should be set to the same value
        var zoom = d3.behavior.zoom().translate([300,300]).scale(.1,.1);
        svg = d3.select('svg').append('svg')
            .attr('y',500)
            .attr('width', width)
            .attr('height', height)
            .call(zoom.on('zoom', zoomed))
            .on('dblclick.zoom', null)
            .append('g')
            .attr('transform', 'translate(300,300)scale(.1,.1)');

        //Set up tooltip
        var tip = d3.tip()
            .attr('class', 'd3-tip')
            .offset([-10, 0])
            .html(function(d) {
                name = d.name;
                tooltip_label = d.name;
                return '<strong>' + tooltip_label + "</strong>";
            })
        svg.call(tip);

        // to prevent excessive animation
        svg.on('mouseup', function() {
              // force.stop();
            });

        var link = svg.selectAll('.link'),
            node = svg.selectAll('.node');

        // ***************************************************************************
        // Run force simulation for n steps then stop it
        // ***************************************************************************

        n = simulationNumber;
        force
          .nodes(graph.nodes)
          .links(graph.links)
          .start();
        for (var i = n * n; i > 0; --i) {
          force.tick();
        }
        setTimeout(function(){ force.stop(); }, simulationTimeout * 1000);

        // ***************************************************************************
        // For fading the currently selected nodes
        // ***************************************************************************

        var linkedByIndex = {};
        for (i = 0; i < graph.nodes.length; i++) {
            linkedByIndex[i + ',' + i] = 1;
        };
        graph.links.forEach(function(d) {
            linkedByIndex[d.source.index + ',' + d.target.index] = 1;
        });
        function neighbouring(a, b) {
            return linkedByIndex[a.index + ',' + b.index];
        }
        function isConnected(a, b) {
            return neighbouring(a, b) || neighbouring(b, a);
        }
        selectNode = function(d) {
            if (toggle == 0) {

                toggle = 1;

                d = d3.select(this).node().__data__;
                if(d.is_topic) {
                    $('#message').text('Loading ' + d.name)
                    $('#message').fadeIn('fast');
                    load_parents(d.node_id,d.name,vo_id);
                    plot_word_graph('/basicviz/get_word_graph/'+d.node_id, d.node_id, d.name);
                    plot_word_graph('/basicviz/get_intensity/'+d.node_id, d.node_id, d.name);
                }

                // reduce the opacity of all but the neighbouring nodes
                node.style('opacity', function(o) {
                    return isConnected(d, o) ? 1.0 : 0.1;
                });
                link.style('opacity', function(o) {
                    return d.index==o.source.index | d.index==o.target.index ? 1.0 : 0.1;
                });
                graph.nodes.forEach(function(o) {
                    if (isConnected(d, o)) {
                        if ( (isTopicNode(d) && !isTopicNode(o)) || (!isTopicNode(d) && isTopicNode(o)) ) {
                            target = document.getElementById(o.name + '_label');
                            target.style.display = 'inline';
                        }
                    }
                });

            } else {
                unselectNode(d);
            }

        }

        unselectNode = function(d) {
            // restore opacity
            node.style('opacity', 1);
            link.style('opacity', 1);
            text.style('display', 'none');
            toggle = 0;
            // remove parent plot svg ??
            // d3.select('#frag_graph_svg').remove()
        }

        function setNodeColour(d) {
            if (d.special == true) {
                if (d.hasOwnProperty('highlight_colour')) {
                    return d.highlight_colour; // returns the user-defined highlight colour
                } else {
                    return specialNodeColour; // returns the default colour for highlighted item
                }
            } else {
                if (isNumber(d.score) && d.score >= 0) {
                    return color(d.score);
                } else {
                    return defaultNodeColour;
                }
            }
        }

        function setNodeSize(d) {
            return 10 * Math.pow(size(d.size), 2);
        }

        // ***************************************************************************
        // Setup links and nodes in the graph
        // ***************************************************************************

        var link = link.data(graph.links)
            .enter().append('line')
                .attr('class', 'link')
                .style("stroke-width", function(d) { return d.weight; });

        var node = node.data(graph.nodes)
            .enter().append('g')
                .call(drag)
                .on('dblclick', selectNode)
                .on('mouseover', tip.show)
                .on('mouseout', tip.hide)

        var circle = node.append('path')
            .attr('class', 'node-shape')
            .attr('d',d3.svg.symbol()
                .size(setNodeSize)
                .type(function(d) { return d.type; })
            )
            .attr('id', function(d) { return d.name + '_circle'; })
            .style('fill', setNodeColour);

        var text = node.append('text')
            .attr('class', 'node-label')
            .attr('dx', 10)
            .attr('dy', '.35em')
            .attr('id', function(d) { return d.name + '_label'; })
            .style('display', 'none') // initially all node labels are invisible
            .text(function(d) { return '\u2002' + d.name; });

        // for the search box
        for (var i = 0; i < graph.nodes.length - 1; i++) {
            optArray.push(graph.nodes[i].name);
        }
        optArray = optArray.sort();

        // *****************************************************************************
        // Handle force tick event
        // *****************************************************************************

        function tick() {

            link.attr('x1', function(d) { return d.source.x; })
                .attr('y1', function(d) { return d.source.y; })
                .attr('x2', function(d) { return d.target.x; })
                .attr('y2', function(d) { return d.target.y; });

            node.attr('transform', function(d) { return 'translate(' + d.x + ',' + d.y + ')'; });

        }

        // *****************************************************************************
        // Fix nodes positions after double click or dragging
        // *****************************************************************************

        function dblclick(d) {
            d3.select(this).classed('fixed', d.fixed = false);
        }

        function dragstart(d) {
            d3.event.sourceEvent.stopPropagation();
            // d3.select(this).classed('fixed', d.fixed = true);
        }

        // *****************************************************************************
        // For zooming
        // *****************************************************************************

        function zoomed() {
            svg.attr('transform',
                'translate(' + d3.event.translate + ') scale(' + d3.event.scale + ')');
        }

        // *****************************************************************************
        // Key handler
        // *****************************************************************************

        var keyh = true;
        d3.select(window).on("keydown", keydown);

        function keydown() {
            if (d3.event.keyCode == 32) {
                force.stop();
            } else if (d3.event.keyCode >= 48 && d3.event.keyCode <= 90 &&
                    !d3.event.ctrlKey && !d3.event.altKey && !d3.event.metaKey) {
                pressed = String.fromCharCode(d3.event.keyCode);
                switch (pressed) {
                    case "H":

                        keyh = !keyh;

                        console.log('keypressed = ' + pressed + ' keyh=' + keyh);
                        nodes_to_hide = node.filter(function(d) {
                            if (isTopicNode(d)) {
                                return !d.special; // don't hide special topic nodes
                            } else {
                                // don't hide document connected to a special topic node
                                var count = 0;
                                graph.nodes.forEach(function(o) {
                                    if (isConnected(d, o) && o.special && isTopicNode(o)) {
                                        count += 1;
                                    }
                                });
                                return count > 0 ? false : true;
                            }
                        });
                        links_to_hide = link.filter(function(d) {
                            if ( (d.source.special && isTopicNode(d.source)) ||
                                (d.target.special && isTopicNode(d.target)) ){
                                // keep if either side is a special topic node
                                return false;
                            }
                            return true;
                        });
                        if (keyh) {
                            node.style('display', 'inline');
                            link.style('display', 'inline');
                        } else {
                            nodes_to_hide.style('display', 'none');
                            links_to_hide.style('display', 'none');
                        }

                        break;

                }
            }
        }

        // *****************************************************************************
        // Other functions
        // *****************************************************************************

        function isNumber(n) {
            return !isNaN(parseFloat(n)) && isFinite(n);
        }

        function isTopicNode(d) {
            if (d.name.indexOf(topicName) > -1) {
                return true;
            } else {
                return false;
            }
        }

        msg = document.getElementById('message');
        msg.innerHTML = 'Search for a node above.<br/>' +
            '<b>Double-click</b> on a Mass2Motif in the network graph to select it.<br/>' +
            'Press [H] to show only the special nodes';
        setTimeout(function() {
            $('#message').fadeOut('fast');
        }, 10000);

    });

    // *****************************************************************************
    // Search node
    // *****************************************************************************

    $(function () {
        $("#searchText").autocomplete({
            source: optArray
        });
    });

    function searchNode() {

        //find the node
        var selectedVal = document.getElementById('searchText').value;
        if (selectedVal == "none") {
            node.style("stroke", "white").style("stroke-width", "1");
        } else {
            var selected = graphNodes.filter(function (d, i) {
                return d.name == selectedVal;
            });
            toggle = 0;
            selected.each(function(d, i) {
                var onClickFunc = d3.select(this).on("dblclick");
                onClickFunc.apply(this, [d, i]);
            });
        }

    }

    function resetSearch() {
        var searchText = document.getElementById('searchText');
        searchText.value = '';
        unselectNode();
    }

} //end plot_graph()