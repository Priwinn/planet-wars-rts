package games.planetwars.runners

import competition_entry.GreedyHeuristicAgent
import games.planetwars.agents.PlanetWarsAgent
import games.planetwars.agents.RemoteAgent
import games.planetwars.agents.evo.SimpleEvoAgent
import games.planetwars.core.GameParams
import json_rmi.SimpleAgent
import java.io.File

fun main(args: Array<String>) {
    if (args.isEmpty()) {
        println("❌ Please provide the port number for the remote agent.")
        return
    }

    val remotePort = args[0].toIntOrNull()
    if (remotePort == null) {
        println("❌ Invalid port number: ${args[0]}")
        return
    }

    val gameParams = GameParams(numPlanets = 20, maxTicks = 2000)
    val baselineAgents = SamplePlayerLists().getRandomTrio()
    baselineAgents.add(GreedyHeuristicAgent())
    baselineAgents.add(SimpleEvoAgent())
    val remoteAgent = RemoteAgent("<unused - name retrieved from remoteAgent>", port = remotePort)
    val testAgentName = remoteAgent.getAgentType()
    val results = mutableListOf<Triple<String, Double, Int>>()

    for (baseline in baselineAgents) {
        println("Running $testAgentName against sample: ${baseline.getAgentType()}... ")
        val league = RoundRobinLeague(
            agents = listOf(remoteAgent, baseline),
            gameParams = gameParams,
            gamesPerPair = 5,
            runRemoteAgents = true,
            timeout = 10,
        )

        val scores = league.runRoundRobin()
        val testEntry = scores[testAgentName]
        if (testEntry != null) {
            results.add(Triple(baseline.getAgentType(), testEntry.winRate(), testEntry.nGames))
        }
    }

    val totalPoints = results.sumOf { it.second * it.third / 100.0 }
    val totalGames = results.sumOf { it.third }
    val avgWinRate = if (totalGames > 0) (100 * totalPoints / totalGames) else 0.0

    val markdown = buildString {
        append("### $testAgentName Evaluation\n\n")
        append("| Opponent | Win Rate % | Games Played |\n")
        append("|----------|------------|---------------|\n")
        for ((opponent, winRate, games) in results) {
            append("| $opponent | ${"%.1f".format(winRate)} | $games |\n")
        }
        append("| **Overall Average** | **${"%.1f".format(avgWinRate)}** | **$totalGames** |\n\n")
        append("AVG=${"%.1f".format(avgWinRate)}\n")
    }

    val outputDir = File("results/sample")
    outputDir.mkdirs()
    val outputFile = File(outputDir, "league.md")
    outputFile.writeText(markdown)

    println(markdown)
}
