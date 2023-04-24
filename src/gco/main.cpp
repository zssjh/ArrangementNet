#include "gco/GCoptimization.h"

#include <vector>
#include <map>
#include <memory>
#include <set>
#include <queue>
#include <unordered_set>
#include <unordered_map>
#include <fstream>
#include <cmath>
extern "C" {
struct MySmoothCostFnFromArray : public GCoptimization::SmoothCostFunctor {
    using D = double;
    using I = int;
    MySmoothCostFnFromArray(
        std::map<std::pair<int, int>, double>& cost)
        : cost_(cost)
    {}
    D compute(I s1, I s2, int l1, int l2)
    {
        if (l1 == l2) return 0;
        return cost_[std::make_pair(s1, s2)];
    }
    std::map<std::pair<int, int>, double>& cost_;
};

void process(int nNodes, int nEdges, int nVertices, int nE2E,
    double* nodeScore, double* edgeScore,
    int* srcIdx, int* dstIdx, int* oNodeLabel, int* oEdgeLabel, int* oHEMap,
    int* faceIndices, int* edgeIndices, int* srcE2eIdx, int* dstE2eIdx, double* V,
    double floorWallRatio,
    double extendRatio,
    double edgeIncrease)
{
    for (int i = 0; i < nEdges; i++) {
        edgeScore[i] += edgeIncrease;
        if (edgeScore[i] > 1)
            edgeScore[i] = 1.0;
    }
    auto makeEdge = [&](int x) {
        int v0 = edgeIndices[x * 2];
        int v1 = edgeIndices[x * 2 + 1];
        return (v0 < v1) ? std::make_pair(v0, v1) : std::make_pair(v1, v0);
    };

    std::vector<double> vertex_label_costs(nNodes * 2);
    for (int i = 0; i < nNodes; i++) {
        vertex_label_costs[i * 2] = floorWallRatio * nodeScore[i];
        vertex_label_costs[i * 2 + 1] = floorWallRatio * (1 - nodeScore[i]);
    }
    auto gc = std::make_shared<GCoptimizationGeneralGraph>(nNodes, 2);
    gc->setDataCost(vertex_label_costs.data());
    std::map<std::pair<int, int>, double> cost;
    for (int i = 0; i < nEdges; i++) {
        gc->setNeighbors(srcIdx[i], dstIdx[i], 1);
        cost[std::make_pair(srcIdx[i], dstIdx[i])] = 1 - edgeScore[i];
    }
    MySmoothCostFnFromArray smoothFunc(cost);
    gc->setSmoothCostFunctor(&smoothFunc);
    for (int i = 0; i < nNodes; i++) {
        gc->setLabel(GCoptimization::SiteID(i), GCoptimization::LabelID(nodeScore[i] > 0.5));
    }
    constexpr int kABExpansion = 10;
    (void)gc->expansion(kABExpansion);
    std::set<std::pair<int, int>> validedges;
    for (int i = 0; i < nNodes; i++) {
        oNodeLabel[i] = gc->whatLabel(i);
        if (oNodeLabel[i] > 0) {
            for (int j = 0; j < 3; j++) {
                int v0 = faceIndices[i * 3 + j];
                int v1 = faceIndices[i * 3 + (j + 1) % 3];
                if (v0 > v1) std::swap(v0, v1);
                validedges.insert(std::make_pair(v0, v1));
            }
        }
    }
    for (int i = 0; i < nEdges; i++) {
        oEdgeLabel[i] = 0;
    }
    for (int i = 0; i < nEdges; i++) {
        if (oNodeLabel[srcIdx[i]] != oNodeLabel[dstIdx[i]]) {
            oEdgeLabel[i] = 1;
        } else {
            if (edgeScore[i] > 0.5 && validedges.count(makeEdge(i)))
                oEdgeLabel[i] = 2;
        }
    }
    std::vector<std::unordered_set<int>> occGraph(nVertices);
    for (int i = 0; i < nEdges; i++) {
        if (oEdgeLabel[i] == 1) {
            int idx1 = edgeIndices[i * 2];
            int idx2 = edgeIndices[i * 2 + 1];
            occGraph[idx1].insert(idx2);
            occGraph[idx2].insert(idx1);
        }
    }
    std::vector<int> degrees(nVertices, 0);
    for (int i = 0; i < occGraph.size(); ++i) {
        degrees[i] = occGraph[i].size();
    }
    std::vector<std::unordered_set<int>> edgeGraph(nEdges / 2);
    for (int i = 0; i < nE2E / 2; ++i) {
        edgeGraph[srcE2eIdx[i]].insert(dstE2eIdx[i]);
        edgeGraph[dstE2eIdx[i]].insert(srcE2eIdx[i]);
    }
    std::vector<std::set<std::pair<int, int>>> strips;
    while (true) {
        std::map<std::pair<int, int>, int> visitedEdges;
        bool update = false;
        for (int i = 0; i < nEdges / 2; ++i) {
            if (oEdgeLabel[i] != 2) {
                continue;
            }
            auto k = makeEdge(i);
            if (visitedEdges.count(k)) {
                continue;
            }
            visitedEdges[k] = i;
            std::set<std::pair<int, int>> strip;
            strip.insert(k);
            std::queue<int> q;
            q.push(i);
            while (!q.empty()) {
                int eid = q.front();
                q.pop();
                for (auto& nEid : edgeGraph[eid]) {
                    auto nk = makeEdge(nEid);
                    if (visitedEdges.count(nk) || oEdgeLabel[nEid] != 2)
                        continue;
                    q.push(nEid);
                    visitedEdges[nk] = nEid;
                    strip.insert(nk);
                }
            }
            if (extendRatio < 0) {
                strips.push_back(strip);
                continue;
            }
            auto originStrip = strip;
            std::unordered_map<int, int> innerDegrees;
            for (auto& s : strip) {
                innerDegrees[s.first] += 1;
                innerDegrees[s.second] += 1;
            }
            std::queue<std::pair<int, int>> qinfo;
            for (auto& s : strip) {
                if (innerDegrees[s.first] == 1 || innerDegrees[s.second] == 1) {
                    qinfo.push(std::make_pair(visitedEdges[s], 0));
                }
            }
            while (!qinfo.empty()) {
                auto eidInfo = qinfo.front();
                int eid = eidInfo.first;
                int token = eidInfo.second;
                qinfo.pop();
                auto k = makeEdge(eid);
                std::unordered_set<int> extendV;
                if (innerDegrees[k.first] == 1) {
                    extendV.insert(k.first);
                }
                if (innerDegrees[k.second] == 1) {
                    extendV.insert(k.second);
                }
                for (auto& nEid : edgeGraph[eid]) {
                    auto nk = makeEdge(nEid);
                    if (!extendV.count(nk.first) && !extendV.count(nk.second)) continue;
                    if (strip.count(nk)) continue;
                    if (oEdgeLabel[nEid] == 0 && token == 1) continue;
                    int ntoken = oEdgeLabel[nEid];
                    if (ntoken == 0 && ((extendV.count(nk.first) && !occGraph[nk.first].empty())
                        || (extendV.count(nk.second) && !occGraph[nk.second].empty())))
                            continue;
                    innerDegrees[nk.first] += 1;
                    innerDegrees[nk.second] += 1;
                    visitedEdges[nk] = nEid;
                    strip.insert(nk);
                    qinfo.push(std::make_pair(nEid, ntoken));
                }
            }

            double len = 0, lenSolid = 0;
            for (auto& e : strip) {
                double dx = V[e.first * 2] - V[e.second * 2];
                double dy = V[e.first * 2 + 1] - V[e.second * 2 + 1];
                double l = sqrt(dx * dx + dy * dy);
                len += l;
                if (oEdgeLabel[visitedEdges[e]] > 0)
                    lenSolid += l;
            }
            if (lenSolid / len > extendRatio) {
                innerDegrees.clear();
                for (auto& s : strip) {
                    innerDegrees[s.first] += 1;
                    innerDegrees[s.second] += 1;
                }
                bool singularity = false;
                for (auto& s : innerDegrees) {
                    if (s.second == 1 && occGraph[s.first].empty()) {
                        singularity = true;
                    }
                }
                if (!singularity) {
                    strips.push_back(strip);
                    for (int i = 0; i < nEdges; ++i) {
                        auto e = makeEdge(i);
                        if (strip.count(e) && oEdgeLabel[i] != 1 && oEdgeLabel[i] != 3) {
                            occGraph[e.first].insert(e.second);
                            occGraph[e.second].insert(e.first);
                            update = true;
                            oEdgeLabel[i] = 3;
                        }
                    }
                }
            }
        }
        if (!update) {
            break;
        }
    }

    std::map<std::pair<int, int>, int> edge2dedge;
    std::vector<int> E2E(nNodes * 3, -1);
    for (int i = 0; i < nNodes; ++i) {
        for (int j = 0; j < 3; ++j) {
            int v0 = faceIndices[i * 3 + j];
            int v1 = faceIndices[i * 3 + (j + 1) % 3];
            edge2dedge[std::make_pair(v0, v1)] = i * 3 + j;
        }
    }
    for (auto& info : edge2dedge) {
        int v0 = info.first.first;
        int v1 = info.first.second;
        if (edge2dedge.count(std::make_pair(v1, v0))) {
            E2E[info.second] = edge2dedge[std::make_pair(v1, v0)];
        }
    }

    std::vector<double> faceArea(nNodes, 0);
    for (int i = 0; i < nNodes; ++i) {
        double* p0 = V + faceIndices[i * 3] * 2;
        double* p1 = V + faceIndices[i * 3 + 1] * 2;
        double* p2 = V + faceIndices[i * 3 + 2] * 2;
        double dx01 = p1[0] - p0[0];
        double dy01 = p1[1] - p0[1];
        double dx02 = p2[0] - p0[0];
        double dy02 = p2[1] - p0[1];
        double area = std::abs(dx01 * dy02 - dy01 * dx02) * 0.5;
        faceArea[i] = area;
    }
    int groupId = 0;
    std::vector<int> group(nNodes, -1);
    std::vector<double> groupArea;
    for (int i = 0; i < nNodes; ++i) {
        if (oNodeLabel[i] == 0 || group[i] != -1) continue;
        std::queue<int> q;
        q.push(i);
        group[i] = groupId;
        double area = faceArea[i];
        while (!q.empty()) {
            int f = q.front();
            q.pop();
            for (int j = 0; j < 3; ++j) {
                int de = f * 3 + j;
                int v0 = faceIndices[de];
                int v1 = faceIndices[f * 3 + (j + 1) % 3];
                if (occGraph[v0].count(v1)) continue;
                if (E2E[de] == -1) continue;
                int nf = E2E[de] / 3;
                if (oNodeLabel[nf] == 0) continue;
                if (group[nf] >= 0) continue;
                q.push(nf);
                group[nf] = groupId;
                area += faceArea[nf];
            }
        }
        groupArea.push_back(area);
        groupId += 1;
    }
    for (int i = 0; i < groupArea.size(); ++i) {
        if (groupArea[i] < 1) {
            for (int j = 0; j < group.size(); ++j) {
                if (group[j] == i) {
                    oNodeLabel[j] = 2;
                }
            }
        }
    }
    validedges.clear();
    for (int i = 0; i < nNodes; ++i) {
        if (oNodeLabel[i] == 1) {
            for (int j = 0; j < 3; ++j) {
                int v0 = faceIndices[i * 3 + j];
                int v1 = faceIndices[i * 3 + (j + 1) % 3];
                if (v0 > v1) std::swap(v0, v1);
                validedges.insert(std::make_pair(v0, v1));
            }
        }
    }

    for (int i = 0; i < nEdges; ++i) {
        if (oEdgeLabel[i] >= 2) {
            oEdgeLabel[i] -= 2;
        }
        if (!validedges.count(makeEdge(i)) && oEdgeLabel[i] == 1) {
            oEdgeLabel[i] = 2;
        }
    }

    for (int i = 0; i < nEdges / 2; ++i) {
        if (oEdgeLabel[i] != oEdgeLabel[i + nEdges / 2]) {
            printf("OMG! %d %d\n", oEdgeLabel[i], oEdgeLabel[i + nEdges /2]);
            exit(0);
        }
    }
    std::map<std::pair<int, int>, int> edgeLabelmap;
    for (int i = 0; i < nEdges; ++i) {
        auto e  = makeEdge(i);
        edgeLabelmap[e] = i;
    }
    for (int i = 0; i < nNodes; ++i) {
        for (int j = 0; j < 3; ++j) {
            int v0 = faceIndices[i * 3 + j];
            int v1 = faceIndices[i * 3 + (j + 1) % 3];
            if (v0 > v1) std::swap(v0, v1);
            oHEMap[i * 3 + j] = edgeLabelmap[std::make_pair(v0, v1)];
        }
    }
}

};