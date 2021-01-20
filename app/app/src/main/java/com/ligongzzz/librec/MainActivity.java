package com.ligongzzz.librec;

import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.view.View;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;
import com.google.android.material.bottomnavigation.BottomNavigationView;
import androidx.appcompat.app.AppCompatActivity;
import androidx.navigation.NavController;
import androidx.navigation.Navigation;
import androidx.navigation.ui.AppBarConfiguration;
import androidx.navigation.ui.NavigationUI;
import com.ligongzzz.librec.ui.home.HomeFragment;
import org.json.JSONObject;

public class MainActivity extends AppCompatActivity {

    String detail = "";
    int downloadsta = 1;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        BottomNavigationView navView = findViewById(R.id.nav_view);
        // Passing each menu ID as a set of Ids because each
        // menu should be considered as top level destinations.
        AppBarConfiguration appBarConfiguration = new AppBarConfiguration.Builder(
                R.id.navigation_home, R.id.navigation_dashboard, R.id.navigation_notifications)
                .build();
        NavController navController = Navigation.findNavController(this, R.id.nav_host_fragment);
        NavigationUI.setupActionBarWithNavController(this, navController, appBarConfiguration);
        NavigationUI.setupWithNavController(navView, navController);
    }

    public void onLoginButtonClicked(View view) {
        String UserID = ((EditText)findViewById(R.id.loginUserID)).getText().toString();
        String Password = ((EditText)findViewById(R.id.loginPassword)).getText().toString();

        if(UserID.isEmpty() || Password.isEmpty()) {
            Toast.makeText(this, "请输入用户名与密码！", Toast.LENGTH_SHORT).show();
            return;
        }
        
        SharedPreferences sharedPreferences = getSharedPreferences("data", Context.MODE_PRIVATE);
        String IP = sharedPreferences.getString("IP", "[2001:da8:270:2020:f816:3eff:fef7:817c]");

        getHTMLData(IP, UserID, Password);
    }

    public void onIPSetButtonClicked(View view) {
        String IP = ((EditText)findViewById(R.id.IP)).getText().toString();
        SharedPreferences sharedPreferences = getSharedPreferences("data", Context.MODE_PRIVATE);
        SharedPreferences.Editor editor = sharedPreferences.edit();

        editor.putString("IP", IP);
        editor.commit();
        ((TextView)findViewById(R.id.IPShowTextView)).setText("当前IP："+IP);
    }

    private Handler loginHandler = new Handler() {
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
                    int result = jsonObject.getInt("result");

                    if (result==0){
                        Toast.makeText(MainActivity.this, "无效的用户名或密码", Toast.LENGTH_SHORT).show();
                    }
                    else{
                        String token = jsonObject.getString("token");
                        SharedPreferences sharedPreferences = getSharedPreferences("data", Context.MODE_PRIVATE);
                        SharedPreferences.Editor editor = sharedPreferences.edit();

                        editor.putString("token", token);
                        editor.putString("UserID", ((EditText)findViewById(R.id.loginUserID)).getText().toString());
                        editor.commit();
                        Toast.makeText(MainActivity.this,"登录成功！",Toast.LENGTH_SHORT).show();
                    }
                }
                catch (Exception e) {
                    e.printStackTrace();
                    downloadsta = 0;
                }
            }

            if (downloadsta == 0) {
                Toast.makeText(MainActivity.this, "加载发生错误", Toast.LENGTH_SHORT).show();
            }
        }
    };


    public void getHTMLData(final String IP, final String UserID, final String Password) {
        new Thread() {
            public void run() {
                String application = "app_login";
                Handler callbackHandler = loginHandler;

                try {
                    String URL = "http://" + IP + "/"+application;
                    URL += "?UserID=" + UserID + "&Password=" + Password;
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
