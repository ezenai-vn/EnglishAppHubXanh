import { NextRequest, NextResponse } from "next/server";
import OpenAI from "openai";

const groq = new OpenAI({
  apiKey: process.env.GROQ_API_KEY,
  baseURL: "https://api.groq.com/openai/v1",
});

// Helper function to call Gemini 2.5 Flash
async function queryGeminiFlash(prompt: string): Promise<any> {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error("GEMINI_API_KEY is not configured.");
  }
  const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=${apiKey}`;
  const payload = {
    contents: [
      {
        parts: [
          { text: prompt }
        ]
      }
    ],
    generationConfig: {
      responseMimeType: "application/json"
    }
  };

  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(payload)
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Gemini API error: ${response.status} - ${errorText}`);
  }

  const data = await response.json();
  const text = data.candidates?.[0]?.content?.parts?.[0]?.text;
  if (!text) {
    throw new Error("Gemini response is empty.");
  }

  return JSON.parse(text);
}

export async function POST(req: NextRequest) {
  try {
    const formData = await req.formData();
    const audioFile = formData.get("audio") as File | null;
    const textInput = formData.get("text") as string | null;
    const stage = formData.get("stage") as string;
    const chatHistoryRaw = formData.get("chatHistory") as string || "[]";
    const contextRaw = formData.get("context") as string || "{}";
    const mode = formData.get("mode") as string || "practice"; // test vs practice

    const chatHistory = JSON.parse(chatHistoryRaw);
    const context = JSON.parse(contextRaw);

    let transcribedText = textInput || "";

    // 1. Transcribe audio if provided
    if (audioFile && audioFile.size > 0) {
      const audioBuffer = Buffer.from(await audioFile.arrayBuffer());
      const file = new File([audioBuffer], "audio.webm", { type: "audio/webm" });
      
      const transcription = await groq.audio.transcriptions.create({
        model: "whisper-large-v3",
        file: file,
        language: "en",
      });
      transcribedText = (transcription.text || "").trim();
      console.log(`📝 [Groq Whisper] Transcribed text: "${transcribedText}"`);
    }

    if (!transcribedText && chatHistory.length > 0) {
      return NextResponse.json({ error: "Không nhận diện được giọng nói." }, { status: 400 });
    }

    // 2. Calculate reading accuracy if in reading stage
    let readingAccuracy = 100;
    if (stage === "reading" && transcribedText) {
      const referenceStory = context.referenceStory || "Max is a happy little monkey who lives in a very tall coconut tree in the jungle. He loves to eat sweet yellow bananas every morning. Today, Max looks down and sees a small green frog sitting on a leaf in the pond. The frog is jumping up and down and singing a funny song. Max waves hello and laughs happily!";
      const storyWords = referenceStory.toLowerCase().replace(/[.,\/#!$%\^&\*;:{}=\-_`~()?]/g, "").split(/\s+/).filter(Boolean);
      const spokenWords = transcribedText.toLowerCase().replace(/[.,\/#!$%\^&\*;:{}=\-_`~()?]/g, "").split(/\s+/).filter(Boolean);
      
      let matchedCount = 0;
      const tempSpoken = [...spokenWords];
      storyWords.forEach((word: string) => {
        const index = tempSpoken.indexOf(word);
        if (index !== -1) {
          matchedCount++;
          tempSpoken.splice(index, 1);
        }
      });
      readingAccuracy = Math.round((matchedCount / storyWords.length) * 100);
      console.log(`🎯 [Reading Accuracy] Matched ${matchedCount}/${storyWords.length} words. Accuracy: ${readingAccuracy}%`);
    }

    // 3. Construct prompt for Gemini 2.5 Flash
    let stageInstructions = "";
    
    if (stage === "warmup") {
      const aiMessageCount = chatHistory.filter((m: any) => m.role === "ai" || m.role === "assistant").length;
      stageInstructions = `
We are in the Warm-up stage.
The child's response: "${transcribedText}".
Number of AI messages sent in warmup so far: ${aiMessageCount}.

Follow this exact flow:
1. If number of AI messages sent so far is 0: Greet the child warmly, comment on their name, and ask: "How old are you?"
   Set "stageComplete" to false.
2. If number of AI messages sent so far is 1: Praise the child's age, and ask: "What is your favorite animal?"
   Set "stageComplete" to false.
3. If number of AI messages sent so far is 2 or more: Praise the child's favorite animal, and complete Warm-up by returning EXACTLY this sentence in "aiResponse": "Great job! Let's look at a picture now."
   Set "stageComplete" to true.`;
    } else if (stage === "picture") {
      const pictureIndex = context.pictureIndex || 0;
      const subQuestionIndex = typeof context.subQuestionIndex === "number" ? context.subQuestionIndex : 0;
      const questions = context.questions || [];

      stageInstructions = `
We are in the Picture Description stage for Picture ${pictureIndex + 1}.
Here is the questions array for this picture:
${JSON.stringify(questions)}

The child is currently at question index: ${subQuestionIndex}.
Child's response: "${transcribedText}".

Your tasks:
1. Check if the child's response answers the question at index ${subQuestionIndex} (compare against expectedKeywords and targetGrammar semantically).
2. Check if the child's response also answers any of the subsequent questions (indices ${subQuestionIndex + 1}, ${subQuestionIndex + 2}, etc.) in the questions array (this is "real-time pacing" / answering questions in advance).
3. Identify all questions from index ${subQuestionIndex} onwards that the child has successfully answered in this turn.
4. Output their indices in the "answeredIndices" array (e.g. [0] or [0, 1]).
5. Collect all keywords that were matched in the child's response from the expectedKeywords lists of the answered questions. Output them in the "keywordsHit" array. Matches can be semantic or word-level.
6. Determine the "nextSubQuestionIndex": the index of the first unanswered question (e.g. if current is 0 and the child answered 0 and 1, next is 2).
7. If all questions in the array have been answered (meaning nextSubQuestionIndex >= questions.length), set "stageComplete" to true.
8. Formulate a cute, encouraging examiner comment (1-2 sentences with emojis) in "aiResponse":
   - If stageComplete is false: congratulate/praise the child's answer and then ask the question at questions[nextSubQuestionIndex].examinerScript.
   - If stageComplete is true:
     - If pictureIndex is 0: the response MUST end with exactly: "Great job with the first picture! Now let's look at a second picture."
     - If pictureIndex is 1: the response MUST end with exactly: "Excellent! You did a great job with both pictures. Now, let's read a short story together."`;
    } else if (stage === "reading") {
      stageInstructions = `
The child has finished reading the story aloud.
1. Praise the child's reading skills warmly.
2. The response MUST end with exactly: "Fantastic reading! You read the story beautifully. Let's answer a quick question about it now!"
3. Set "stageComplete" to true.`;
    } else {
      stageInstructions = `
The test is ending.
1. Praise the child for their hard work.
2. Say: "You did amazingly well today! Goodbye and see you next time!"
3. Set "stageComplete" to true.`;
    }

    const geminiPrompt = `You are a friendly, encouraging Cambridge YLE (Young Learners English) examiner.
Your job is to talk to a primary student (6-10 years old) in simple English, using short sentences (1-2 sentences) and fun emojis.

Current stage of the exam: "${stage}"
Interactive Mode: "${mode}" (practice/test)
- In "practice" mode, be extra conversational, warm, and guiding.
- In "test" mode, be standard, structured, and examiner-like.

Chat History so far:
${JSON.stringify(chatHistory)}

Child's latest response: "${transcribedText}"

Stage-specific Instructions:
${stageInstructions}

You MUST return a JSON object with the following fields:
{
  "aiResponse": "Your response to the child in English (1-2 sentences with emojis)",
  "stageComplete": true or false,
  "nextSubQuestionIndex": number (only applicable for "picture" stage),
  "answeredIndices": [number] (indices of questions answered in this turn, only for "picture" stage),
  "keywordsHit": ["keyword1", "keyword2"] (list of expected keywords that were found/matched in the child's response, only for "picture" stage)
}`;

    console.log("🤖 [Gemini 2.5 Flash] Querying Gemini model for interactive-chat...");
    const parsedData = await queryGeminiFlash(geminiPrompt);
    console.log("✅ [Gemini 2.5 Flash] Response parsed:", parsedData);

    let finalNextSubQuestionIndex = typeof parsedData.nextSubQuestionIndex === "number" ? parsedData.nextSubQuestionIndex : undefined;
    let finalStageComplete = parsedData.stageComplete || false;
    let finalAiResponse = parsedData.aiResponse || "";

    if (stage === "warmup" && finalStageComplete) {
      finalAiResponse = "Great job! Let's look at a picture now.";
    } else if (stage === "picture") {
      const pictureIndex = context.pictureIndex || 0;
      const subQuestionIndex = typeof context.subQuestionIndex === "number" ? context.subQuestionIndex : 0;
      const questions = context.questions || [];

      // Out-of-bounds safety check: enforce stage complete when all questions in the array are answered
      let nextIndex = finalNextSubQuestionIndex;
      if (nextIndex === undefined) {
        nextIndex = subQuestionIndex + 1;
      }

      if (nextIndex >= questions.length) {
        finalStageComplete = true;
      }

      // If picture stage is completed, ensure transition prompt is correct and doesn't ask any further question
      if (finalStageComplete) {
        let transitionPrompt = "";
        if (pictureIndex === 0) {
          transitionPrompt = "Great job with the first picture! Now let's look at a second picture.";
        } else {
          transitionPrompt = "Excellent! You did a great job with both pictures. Now, let's read a short story together.";
        }

        const cleanAiResponse = finalAiResponse.trim();
        const hasTransition = cleanAiResponse.toLowerCase().includes("second picture") || cleanAiResponse.toLowerCase().includes("read a short story");

        if (!hasTransition) {
          const sentences = cleanAiResponse.split(/(?<=[.!?])\s+/);
          const nonQuestionSentences = sentences.filter(s => !s.includes("?"));
          const praise = nonQuestionSentences.join(" ");
          finalAiResponse = praise ? `${praise} ${transitionPrompt}` : transitionPrompt;
        }
      }
    } else if (stage === "reading") {
      finalStageComplete = true;
      const transitionPrompt = "Fantastic reading! You read the story beautifully. Let's answer a quick question about it now!";
      const cleanAiResponse = finalAiResponse.trim();
      const hasTransition = cleanAiResponse.toLowerCase().includes("fantastic reading") || cleanAiResponse.toLowerCase().includes("beautifully");

      if (!hasTransition) {
        const sentences = cleanAiResponse.split(/(?<=[.!?])\s+/);
        const nonQuestionSentences = sentences.filter(s => !s.includes("?"));
        const praise = nonQuestionSentences.join(" ");
        finalAiResponse = praise ? `${praise} ${transitionPrompt}` : transitionPrompt;
      }
    } else {
      // For any other final stages (writing spelling etc.)
      finalStageComplete = true;
      if (!finalAiResponse.toLowerCase().includes("goodbye") && !finalAiResponse.toLowerCase().includes("see you")) {
        finalAiResponse = "You did amazingly well today! Goodbye and see you next time!";
      }
    }

    return NextResponse.json({
      success: true,
      transcribedText,
      aiResponse: finalAiResponse,
      stageComplete: finalStageComplete,
      nextSubQuestionIndex: finalNextSubQuestionIndex,
      answeredIndices: Array.isArray(parsedData.answeredIndices) ? parsedData.answeredIndices : undefined,
      keywordsHit: Array.isArray(parsedData.keywordsHit) ? parsedData.keywordsHit : undefined,
      readingAccuracy: stage === "reading" ? readingAccuracy : undefined,
    });

  } catch (error: any) {
    console.error("❌ Lỗi API interactive-chat:", error);
    return NextResponse.json(
      { error: "Đã xảy ra lỗi: " + error.message },
      { status: 500 }
    );
  }
}
