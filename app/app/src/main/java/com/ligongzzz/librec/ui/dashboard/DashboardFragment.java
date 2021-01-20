package com.ligongzzz.librec.ui.dashboard;

import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.*;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.lifecycle.Observer;
import androidx.lifecycle.ViewModelProviders;
import com.ligongzzz.librec.GetData;
import com.ligongzzz.librec.InfoActivity;
import com.ligongzzz.librec.R;
import com.ligongzzz.librec.ui.home.HomeFragment;
import org.json.JSONArray;
import org.json.JSONObject;

import java.util.ArrayList;

public class DashboardFragment extends Fragment {
    String detail;
    int downloadsta = 1;
    ArrayList<News> newsList = new ArrayList<News>();
    ListView listView;

    private DashboardViewModel dashboardViewModel;

    public View onCreateView(@NonNull LayoutInflater inflater,
                             ViewGroup container, Bundle savedInstanceState) {
        dashboardViewModel =
                ViewModelProviders.of(this).get(DashboardViewModel.class);
        View root = inflater.inflate(R.layout.fragment_dashboard, container, false);
        listView = root.findViewById(R.id.recListView1);

        dashboardViewModel.getText().observe(getViewLifecycleOwner(), new Observer<String>() {
            @Override
            public void onChanged(@Nullable String s) {
                getRecommendation();
            }
        });
        return root;
    }

    //Adapter
    class NewsAdapter extends ArrayAdapter<News>
    {
        public NewsAdapter(Context context, ArrayList<News> newsList)
        {
            super(context, R.layout.news_entry, newsList);
        }
        @Override
        public View getView(int position, View convertView, ViewGroup parent)
        {
            News news = getItem(position);
            LayoutInflater inflater = getLayoutInflater();
            View view = inflater.inflate(R.layout.news_entry, null);
            TextView textView1 = (TextView) view.findViewById(R.id.newsentryTextView1);
            TextView textView2 = (TextView) view.findViewById(R.id.newsentryTextView2);

            textView1.setText(news.title);
            textView2.setText(news.author);

            return view;
        }
    }

    //Class News
    class News
    {
        public String title;
        public String author;

        public News(String title, String author)
        {
            this.title = title;
            this.author = author;
        }
    }

    private enum RequestType {
        RECOMMEND, BOOK_DETAIL
    }

    public void getRecommendation() {
        // Check Whether Login
        SharedPreferences sharedPreferences = getContext().getSharedPreferences("data", Context.MODE_PRIVATE);
        String token = sharedPreferences.getString("token", "");

        newsList.clear();
        String IP = sharedPreferences.getString("IP", "[2001:da8:270:2020:f816:3eff:fef7:817c]");

        getHTMLData(IP, RequestType.RECOMMEND, token);
    }


    // Get New Info
    public void bookInfo(String ISBN, String name, String author) {
        SharedPreferences sharedPreferences = getActivity().getSharedPreferences("data", Context.MODE_PRIVATE);
        String IP = sharedPreferences.getString("IP", "[2001:da8:270:2020:f816:3eff:fef7:817c]");

        getHTMLData(IP, RequestType.BOOK_DETAIL, ISBN);
    }

    //Handlers
    private Handler listHandler= new Handler() {
        public void handleMessage(Message msg)
        {
            // Not Login
            if (msg.what == 0x1) {
                newsList.add(0, new News("尚未登录", "点击下方的登录按钮登录后进行操作"));
            }
            else {
                // Has Login
                if (detail.equals("Error")) {
                    downloadsta = 0;
                } else {
                    downloadsta = 1;
                    // Parse JSON
                    try {
                        JSONObject jsonObject = new JSONObject(detail);
                        final int result = jsonObject.getInt("result");
                        if (result == -1) {
                            newsList.add(0, new News("无效的会话密钥", "请您重新登录"));
                        }
                        else if (result == 0) {
                            newsList.add(0, new News("无推荐项目", "您的历史借阅项目过少"));
                        }
                        else if (result == 1) {
                            final JSONArray ISBNList = jsonObject.getJSONArray("isbn");
                            final JSONArray NameList = jsonObject.getJSONArray("name");
                            final JSONArray AuthorList = jsonObject.getJSONArray("author");

                            final int returnLength = ISBNList.length();
                            for (int i = 0; i < returnLength; i++) {
                                newsList.add(i, new News(NameList.getString(i), AuthorList.getString(i)));
                            }

                            listView.setOnItemClickListener(new AdapterView.OnItemClickListener() {
                                @Override
                                public void onItemClick(AdapterView<?> parent, View view, int position, long id) {
                                    try {
                                        bookInfo(ISBNList.getString(position), NameList.getString(position), AuthorList.getString(position));
                                    } catch (Exception e) {
                                        e.printStackTrace();
                                    }
                                }
                            });
                        }
                    } catch (Exception e) {
                        e.printStackTrace();
                        downloadsta = 0;
                    }
                }
                if (downloadsta == 0) {
                    Toast.makeText(getContext(), "加载发生错误", Toast.LENGTH_SHORT).show();
                }
            }

            //Create List
            NewsAdapter adapter=new NewsAdapter(getActivity(), newsList);
            listView.setAdapter(adapter);
        }
    };


    private Handler detailHandler = new Handler() {
        public void handleMessage(Message msg)
        {
            if (detail.equals("Error")){
                downloadsta = 0;
            }
            else{
                downloadsta = 1;
                // Parse JSON
                try {
                    JSONObject jsonObject = new JSONObject(detail);
                    Intent intent = new Intent(getContext(), InfoActivity.class);

                    intent.putExtra("name", jsonObject.getString("name"));
                    intent.putExtra("author", jsonObject.getString("author"));
                    intent.putExtra("code", jsonObject.getString("code"));
                    intent.putExtra("year", jsonObject.getString("year"));
                    intent.putExtra("borrow_cnt", jsonObject.getString("borrow_cnt"));
                    intent.putExtra("publisher", jsonObject.getString("publisher"));

                    startActivity(intent);
                }
                catch (Exception e) {
                    e.printStackTrace();
                    downloadsta = 0;
                }
            }

            if (downloadsta == 0) {
                Toast.makeText(getContext(), "加载发生错误", Toast.LENGTH_SHORT).show();
            }
        }
    };


    public void getHTMLData(final String IP, final RequestType requestType, final String toSend) {
        new Thread() {
            public void run() {
                String application = "";
                Handler callbackHandler = listHandler;

                switch (requestType) {
                    case RECOMMEND:
                        application = "app_recommend";

                        if (toSend.isEmpty()) {
                            callbackHandler.sendEmptyMessage(0x1);
                            return;
                        }
                        break;
                    case BOOK_DETAIL:
                        application = "app_book_detail";
                        callbackHandler = detailHandler;
                        break;
                }

                try {
                    String URL = "http://"+IP+"/"+application;
                    URL += "?test=" + toSend;
                    detail = GetData.getHtml(URL);
                } catch (Exception e) {
                    e.printStackTrace();
                    detail = "Error";
                }
                callbackHandler.sendEmptyMessage(0x0);
            };
        }.start();
    }
}