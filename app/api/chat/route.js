import { NextResponse } from "next/server";
import { Pinecone } from "@pinecone-database/pinecone";
import { GoogleGenerativeAI } from "@google/generative-ai";

const systemPrompt = `
You are an AI assistant designed to help evaluate and provide feedback on teacher performance based on student ratings and comments. Your role is to analyze the data provided, identify patterns and trends, and offer constructive insights to help teachers improve their teaching methods and student engagement.
Key Responsibilities:

Analyze numerical ratings across various teaching aspects (e.g., clarity, engagement, fairness).
Interpret qualitative feedback from student comments.
Identify strengths and areas for improvement in teaching performance.
Provide actionable recommendations for professional development.
Maintain objectivity and fairness in your assessments.
Respect student and teacher privacy by not identifying individuals.
`;

// Initialize Gemini AI
const GOOGLE_API_KEY = process.env.GOOGLE_API_KEY;
if (!GOOGLE_API_KEY) {
  throw new Error("GEMINI_API_KEY is not set in environment variables");
}
const genAI = new GoogleGenerativeAI(GOOGLE_API_KEY);

// Initialize Pinecone
const pc = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY,
});

export async function POST(req) {
    try {
        const data = await req.json();
        const index = pc.Index('rag');

        const text = data[data.length-1].content;
        
        // Use Gemini AI for embeddings
        const embeddingModel = genAI.getGenerativeModel({ model: "embedding-001" });
        let embedding;
        try {
            const embeddingResult = await embeddingModel.embedContent(text);
            embedding = embeddingResult.embedding;
            
            // Ensure embedding is a flat array of numbers
            if (Array.isArray(embedding) && embedding.every(item => typeof item === 'number')) {
                console.log("Embedding is valid");
            } else if (embedding && typeof embedding === 'object' && embedding.values) {
                embedding = embedding.values;
                console.log("Extracted values from embedding object");
            } else {
                throw new Error("Invalid embedding format");
            }
        } catch (error) {
            console.error("Error generating or processing embedding:", error);
            return NextResponse.json({ error: "Failed to generate or process embedding" }, { status: 500 });
        }

        // Query Pinecone
        const queryResponse = await index.query({
            vector: embedding,
            topK: 3,
            includeMetadata: true
        });

        let resultString = 'Returned results:';
        queryResponse.matches.forEach((match) => {
            resultString += `
            Professor: ${match.id}
            Review: ${match.metadata.review}
            Subject: ${match.metadata.subject}
            Stars: ${match.metadata.stars}
            \n\n
            `;
        });

        const lastMessage = data[data.length-1];
        const lastMessageContent = lastMessage.content + resultString;
        const lastDataWithoutLastMessage = data.slice(0, data.length-1);

        // Use Gemini AI for text generation
        const model = genAI.getGenerativeModel({ model: "gemini-pro" });
        
        const chatHistory = [
            {
                role: "user",
                parts: [{ text: systemPrompt }]
            },
            ...lastDataWithoutLastMessage.map(msg => ({
                role: msg.role === "assistant" ? "model" : "user",
                parts: [{ text: msg.content }]
            }))
        ];

        console.log("Chat history:", JSON.stringify(chatHistory, null, 2));

        const chat = model.startChat({
            history: chatHistory,
        });

        const result = await chat.sendMessage(lastMessageContent);
        const response = await result.response;
        const generatedText = response.text();

        return new NextResponse(generatedText);
    } catch (error) {
        console.error("Error in POST request:", error);
        return NextResponse.json({ error: error.message }, { status: 500 });
    }
}