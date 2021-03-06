// Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.yahoo.vespa.hosted.provision.restapi;

import com.yahoo.config.provision.ClusterResources;
import com.yahoo.slime.Cursor;
import com.yahoo.slime.Slime;
import com.yahoo.vespa.hosted.provision.Node;
import com.yahoo.vespa.hosted.provision.NodeList;
import com.yahoo.vespa.hosted.provision.applications.Application;
import com.yahoo.vespa.hosted.provision.applications.Cluster;
import com.yahoo.vespa.hosted.provision.autoscale.AllocatableClusterResources;

import java.net.URI;
import java.util.Collection;
import java.util.List;

/**
 * Serializes application information for nodes/v2/application responses
 */
public class ApplicationSerializer {

    public static Slime toSlime(Application application, List<Node> applicationNodes, URI applicationUri) {
        Slime slime = new Slime();
        toSlime(application, applicationNodes, slime.setObject(), applicationUri);
        return slime;
    }

    private static void toSlime(Application application,
                                List<Node> applicationNodes,
                                Cursor object,
                                URI applicationUri) {
        object.setString("url", applicationUri.toString());
        object.setString("id", application.id().toFullString());
        clustersToSlime(application.clusters().values(), applicationNodes, object.setObject("clusters"));
    }

    private static void clustersToSlime(Collection<Cluster> clusters, List<Node> applicationNodes, Cursor clustersObject) {
        clusters.forEach(cluster -> toSlime(cluster, applicationNodes, clustersObject.setObject(cluster.id().value())));
    }

    private static void toSlime(Cluster cluster, List<Node> applicationNodes, Cursor clusterObject) {
        List<Node> nodes = NodeList.copyOf(applicationNodes).not().retired().cluster(cluster.id()).asList();
        int groups = (int)nodes.stream().map(node -> node.allocation().get().membership().cluster().group()).distinct().count();
        ClusterResources currentResources = new ClusterResources(nodes.size(), groups, nodes.get(0).flavor().resources());

        toSlime(cluster.minResources(), clusterObject.setObject("min"));
        toSlime(cluster.maxResources(), clusterObject.setObject("max"));
        toSlime(currentResources, clusterObject.setObject("current"));
        cluster.suggestedResources().ifPresent(suggested -> toSlime(suggested, clusterObject.setObject("suggested")));
        cluster.targetResources().ifPresent(target -> toSlime(target, clusterObject.setObject("target")));
    }

    private static void toSlime(ClusterResources resources, Cursor clusterResourcesObject) {
        clusterResourcesObject.setLong("nodes", resources.nodes());
        clusterResourcesObject.setLong("groups", resources.groups());
        NodeResourcesSerializer.toSlime(resources.nodeResources(), clusterResourcesObject.setObject("resources"));
    }

}
